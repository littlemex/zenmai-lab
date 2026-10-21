## Purpose

`isaacsim-workstation` コンポーネントに対して実施したサンドボックス検証（ap-northeast-1、g6e.2xlarge）で発見された 6 件の問題を修正します。

- **[CRITICAL]** `TERM` 未設定により `create-isaaclab-conda-env` / `fix-setuptools-and-install-isaac-lab` が常に失敗する
- **[HIGH]** dpkg ロック待機ループの複数バグにより `apt-get` が誤って成功と判定される
- **[HIGH]** README の `cd cdk` ディレクトリ誤記（正: `isaacsim-workstation`）
- **[HIGH]** 必須パラメータ `AllowedCidr` が README の設定表に記載されていない
- **[MEDIUM]** README にデプロイ元インスタンス ID (`i-0986d7fe5a672b6f5`) と `us-east-1` がハードコードされている
- **[MEDIUM]** README のステップ再実行パスが誤記（`/var/lib/dcv-bootstrap/` → `/var/lib/workstation-bootstrap/`）

## Changes

- **`userdata_script.sh`**: `create-isaaclab-conda-env` と `fix-setuptools-and-install-isaac-lab` の両 `sudo -u ubuntu bash -c` ブロック冒頭に `export TERM=xterm` を追加
- **`userdata_script.sh`**: `wait-for-dpkg-lock` ステップを全面改修（`unattended-upgrades` の事前停止、タイムアウト時の明示的失敗、`/var/lib/dpkg/lock` と `lock-frontend` の両方を監視、`apt-get` に `-o DPkg::Lock::Timeout=300` を追加）
- **`isaacsim-workstation/README.md`**: Deploy セクションの `cd cdk` → `cd isaacsim-workstation`
- **`isaacsim-workstation/README.ja.md`**: 同上
- **`isaacsim-workstation/README.md`**: Configuration 表に `AllowedCidr` 行を追加、デプロイコマンド例を更新
- **`isaacsim-workstation/README.ja.md`**: 同上
- **`isaacsim-workstation/README.md`**: `SetPasswordCommand` 例のハードコード済みインスタンス ID とリージョンをプレースホルダに置換
- **`isaacsim-workstation/README.ja.md`**: 同上
- **`isaacsim-workstation/README.md`**: UserData 再実行セクションのパスを `/var/lib/dcv-bootstrap/` → `/var/lib/workstation-bootstrap/` に修正
- **`isaacsim-workstation/README.ja.md`**: 同上

### 変更差分

#### 修正 1: TERM=xterm（Critical）— `userdata_script.sh`

```diff
--- a/isaacsim-workstation/lib/constructs/userdata_script.sh
+++ b/isaacsim-workstation/lib/constructs/userdata_script.sh
@@ -184,6 +184,7 @@ must "create-isaaclab-conda-env" '
   ISAACLAB_DIR="/home/ubuntu/IsaacLab"
   sudo -u ubuntu bash -c "
+    export TERM=xterm
     source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
     if ! conda env list | grep -q env_isaaclab; then
       cd $ISAACLAB_DIR && ./isaaclab.sh --conda
     fi
   "
'

@@ -194,6 +195,7 @@ must "fix-setuptools-and-install-isaac-lab" '
   ISAACLAB_DIR="/home/ubuntu/IsaacLab"
   sudo -u ubuntu bash -c "
+    export TERM=xterm
     source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
     conda activate env_isaaclab
```

#### 修正 2: dpkg ロック待機の強化（High）— `userdata_script.sh`

```diff
--- a/isaacsim-workstation/lib/constructs/userdata_script.sh
+++ b/isaacsim-workstation/lib/constructs/userdata_script.sh
@@ -97,13 +97,23 @@ apt_update() {
-  retry "apt-get update -yq" "apt-get update" 6 8
+  retry "apt-get update -yq -o DPkg::Lock::Timeout=300" "apt-get update" 6 30
 }
 apt_install() {
   local pkgs="$*"
-  retry "apt-get install -yq --no-install-recommends $pkgs" "install: $pkgs" 6 8
+  retry "apt-get install -yq --no-install-recommends -o DPkg::Lock::Timeout=300 $pkgs" "install: $pkgs" 6 30
 }

 # 0) Wait for unattended-upgrades / dpkg lock to be released (up to 5 min)
 must "wait-for-dpkg-lock" '
+  systemctl stop unattended-upgrades 2>/dev/null || true
+  systemctl mask unattended-upgrades 2>/dev/null || true
   for i in $(seq 1 60); do
-    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
+    if ! fuser /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
       log "dpkg lock is free"
       break
     fi
     log "Waiting for dpkg lock (attempt $i/60)..."
     sleep 10
+    if [[ $i -eq 60 ]]; then
+      echo "ERROR: dpkg lock not released after 600 s" >&2
+      exit 1
+    fi
   done
 '
```

#### 修正 3: README `cd cdk` 誤記（High）— `README.md` / `README.ja.md`

```diff
--- a/isaacsim-workstation/README.md
+++ b/isaacsim-workstation/README.md
@@ -41,7 +41,7 @@ You can customize settings in the `config` section of `cdk.json`:
 ## Deploy

 ```bash
-cd cdk
+cd isaacsim-workstation
 npm install
 cdk deploy
 ```
```

```diff
--- a/isaacsim-workstation/README.ja.md
+++ b/isaacsim-workstation/README.ja.md
@@ -41,7 +41,7 @@ NVIDIA Isaac Sim Development Workstation AMI を使用した GPU インスタン
 ## デプロイ

 ```bash
-cd cdk
+cd isaacsim-workstation
 npm install
 cdk deploy
 ```
```

#### 修正 4: AllowedCidr の README 追記（High）— `README.md` / `README.ja.md`

```diff
--- a/isaacsim-workstation/README.md
+++ b/isaacsim-workstation/README.md
@@ -52,6 +52,7 @@ You can customize settings in the `config` section of `cdk.json`:
 | `StackPrefix` | `Dev` | Stack name suffix (`DevWorkstation`) |
 | `VpcId` | `""` (create new) | Specify to use an existing VPC |
 | `SubnetAZ` | `""` (auto-select) | Explicitly specify an AZ |
 | `InstanceType` | `g6e.8xlarge` | GPU instance type |
+| `AllowedCidr` | *(required)* | CIDR block allowed to reach DCV (port 8443) and TensorBoard (port 6006) |

 To specify via command line:

 ```bash
-cdk deploy -c VpcId=vpc-xxxxxxxx -c SubnetAZ=subnet-xxxxxxxx
+cdk deploy -c AllowedCidr=$(curl -s https://checkip.amazonaws.com)/32
 ```
```

```diff
--- a/isaacsim-workstation/README.ja.md
+++ b/isaacsim-workstation/README.ja.md
@@ -52,6 +52,7 @@ NVIDIA Isaac Sim Development Workstation AMI を使用した GPU インスタン
 | `StackPrefix` | `Dev` | スタック名サフィックス（`DevWorkstation`） |
 | `VpcId` | `""` (新規作成) | 既存 VPC を使う場合に指定 |
 | `SubnetAZ` | `""` (自動選択) | AZ を明示的に指定する場合 |
 | `InstanceType` | `g6e.8xlarge` | GPU インスタンスタイプ |
+| `AllowedCidr` | *(必須)* | DCV（8443番ポート）および TensorBoard（6006番ポート）へのアクセスを許可する CIDR |

 コマンドラインで指定する場合:

 ```bash
-cdk deploy -c VpcId=vpc-xxxxxxxx -c SubnetAZ=subnet-xxxxxxxx
+cdk deploy -c AllowedCidr=$(curl -s https://checkip.amazonaws.com)/32
 ```
```

#### 修正 5: ハードコード済みインスタンス ID / リージョンの除去（Medium）— `README.md` / `README.ja.md`

```diff
--- a/isaacsim-workstation/README.md
+++ b/isaacsim-workstation/README.md
@@ -130,7 +130,7 @@ From the PC where you ran `cdk deploy`, set the password for DCV login. Use the
 ```bash
 export UBUNTU_PW="your-password-here"

 # Run the SetPasswordCommand from the stack outputs as-is
-aws ssm send-command --instance-ids i-0986d7fe5a672b6f5 --document-name "AWS-RunShellScript" --parameters "commands=[\"HASHED=\$(openssl passwd -6 '${UBUNTU_PW}') && sudo usermod --password \\\"\$HASHED\\\" ubuntu\"]" --region us-east-1 --output text --query "Command.CommandId"
+aws ssm send-command --instance-ids <instance-id> --document-name "AWS-RunShellScript" --parameters "commands=[\"HASHED=\$(openssl passwd -6 '${UBUNTU_PW}') && sudo usermod --password \\\"\$HASHED\\\" ubuntu\"]" --region <region> --output text --query "Command.CommandId"
 ```
```

```diff
--- a/isaacsim-workstation/README.ja.md
+++ b/isaacsim-workstation/README.ja.md
@@ -130,7 +130,7 @@ cdk deployを実行したPCから、DCV ログイン用のパスワードを設定
 ```bash
 export UBUNTU_PW="your-password-here"

 # スタック出力の SetPasswordCommand をそのまま実行
-aws ssm send-command --instance-ids i-0986d7fe5a672b6f5 --document-name "AWS-RunShellScript" --parameters "commands=[\"HASHED=\$(openssl passwd -6 '${UBUNTU_PW}') && sudo usermod --password \\\"\$HASHED\\\" ubuntu\"]" --region us-east-1 --output text --query "Command.CommandId"
+aws ssm send-command --instance-ids <instance-id> --document-name "AWS-RunShellScript" --parameters "commands=[\"HASHED=\$(openssl passwd -6 '${UBUNTU_PW}') && sudo usermod --password \\\"\$HASHED\\\" ubuntu\"]" --region <region> --output text --query "Command.CommandId"
 ```
```

#### 修正 6: UserData 再実行パスの誤記修正（Medium）— `README.md` / `README.ja.md`

```diff
--- a/isaacsim-workstation/README.md
+++ b/isaacsim-workstation/README.md
@@ -258,7 +258,7 @@ Only failed steps can be re-run (idempotent):
 ```bash
 sudo ls -la /var/lib/workstation-bootstrap/
 # Delete the marker for a specific step and re-run
-sudo rm /var/lib/dcv-bootstrap/install-ros2-jazzy.done
+sudo rm /var/lib/workstation-bootstrap/install-ros2-jazzy.done
 sudo bash /var/lib/cloud/instance/scripts/part-001
 ```
```

```diff
--- a/isaacsim-workstation/README.ja.md
+++ b/isaacsim-workstation/README.ja.md
@@ -258,7 +258,7 @@ Only failed steps can be re-run (idempotent):
 ```bash
 sudo ls -la /var/lib/workstation-bootstrap/
 # 特定ステップのマーカーを削除して再実行
-sudo rm /var/lib/dcv-bootstrap/install-ros2-jazzy.done
+sudo rm /var/lib/workstation-bootstrap/install-ros2-jazzy.done
 sudo bash /var/lib/cloud/instance/scripts/part-001
 ```
```

## Test Plan

**Environment:**

- AWS Service: Amazon EC2
- Instance type: g6e.2xlarge（ap-northeast-1a）
- Number of nodes: 1

**Test commands:**

```bash
# デプロイ（アカウント: <account-id> / ap-northeast-1 / サンドボックス）
cd isaacsim-workstation
npm install
MY_IP=$(curl -s https://checkip.amazonaws.com)
cdk deploy -c InstanceType=g6e.2xlarge -c AllowedCidr=${MY_IP}/32

# UserData 完了まで待機
aws ec2 wait instance-status-ok --instance-ids <instance-id> --region ap-northeast-1

# ブートストラップ結果確認（SSM 経由）
aws ssm start-session --target <instance-id> --region ap-northeast-1
sudo cat /var/log/workstation-bootstrap.summary

# TERM 未設定が根本原因であることの再現確認（修正前の動作）
TERM=unknown bash -c 'set -e; tabs 4'; echo "exit: $?"
# → exit: 1  (TERM=unknown では tabs が非ゼロで終了し set -e がトリガーされる)

# TERM=xterm で同様の確認（修正後の動作）
TERM=xterm bash -c 'set -e; tabs 4'; echo "exit: $?"
# → exit: 0
```

## Test Results

### 修正前

```
STEP_OK:wait-for-dpkg-lock
STEP_OK:install-basic-utils
STEP_OK:install-efs-utils
STEP_WARN:mount-s3files
STEP_OK:install-ros2-jazzy
STEP_OK:install-isaacsim-pip-deps
STEP_OK:install-miniconda
STEP_OK:clone-isaac-lab
STEP_FAIL:create-isaaclab-conda-env:line=189:rc=1:cmd=./isaaclab.sh --conda
```

- `create-isaaclab-conda-env` が `./isaaclab.sh --conda` 実行直後に rc=1 で失敗
- 後続の `fix-setuptools-and-install-isaac-lab` も同様に失敗
- 根本原因特定: `isaaclab.sh` 内 `set -e` 環境下で `tabs 4` が `TERM=unknown`（sudo による TERM 除去）のため終了コード 1 を返す

**再現コマンド:**
```
# インスタンス上で確認
TERM=unknown bash -c 'set -e; tabs 4'
# → exit 1  ← TERM=xterm では exit 0
```

### 修正後

```
STEP_OK:wait-for-dpkg-lock
STEP_OK:install-basic-utils
STEP_OK:install-efs-utils
STEP_WARN:mount-s3files
STEP_OK:install-ros2-jazzy
STEP_OK:install-isaacsim-pip-deps
STEP_OK:install-miniconda
STEP_OK:clone-isaac-lab
STEP_OK:create-isaaclab-conda-env
STEP_OK:fix-setuptools-and-install-isaac-lab
STEP_OK:build-isaacsim-ros2-workspace
```

全 11 ステップが `STEP_OK`（`mount-s3files` のみ S3 Files 未設定環境のため `STEP_WARN`）。

## Checklist

- [x] [コントリビューションガイドライン](https://github.com/awslabs/awsome-distributed-training/blob/main/CONTRIBUTING.md)を読みました。
- [x] 最新の `main` ブランチに対して作業しています。
- [x] 既存のオープン PR および最近マージされた PR を検索し、重複がないことを確認しました。
- [x] このコントリビューションはドキュメントとスクリプトを含む自己完結した内容です。
- [x] 外部依存関係は特定のバージョンまたはタグに固定されています（`latest` は使用していません）。
- [x] 前提条件・手順・既知の問題を含む README を更新しました。
- [x] 新しいテストケースは[期待されるディレクトリ構造](#directory-structure)に従っています（該当なし）。