十分なデータを読み込みました。refuted findings（ボツ finding）のセクションに必要な情報も得られています。以下にレポートを生成します。

---

# PASK (sample-physical-ai-scaffolding-kit) コードレビュー報告書
**対象リポジトリ**: https://github.com/aws-samples/sample-physical-ai-scaffolding-kit
**レビュー範囲**: isaacsim-workstation サブプロジェクト全体（6レンズ, 90件確認済み findings）
**作成日**: 2026-06-05
**宛先**: Akazawa-san (AWS SA)

---

## 1. 全体評価

PASK v0.0.1 は「動く概念実証」として一定の完成度を持つが、プロダクション R&D 用途に直接適用するには修正が必要な状態である。TERM 未設定バグや dpkg ロック競合など初回デプロイで必ず踏む critical/high の機能バグが複数存在し、「設計が雑」というユーザー仮説は概ね正しい。一方、ステップ管理の must() 関数、S3 Files 統合、DCV による GUI 接続など骨格となる設計は筋が通っており、cdk.context.json の漏洩も hyperpod では対処済みであることから「意図的な悪さ」ではなくプレビューリリース特有の品質不足と評価できる。セキュリティ (IMDSv2 未強制, ECR PowerUser 過剰権限, EBS 非暗号化) も実害レベルに昇格しており、バグ修正と並行した対応が求められる。

---

## 2. PR 分割戦略

### 推奨: 複数の小さな PR

**理由**:

1. **マージ速度** — aws-samples は外部 contributor の PR をレビューするコストが小さい PR ほど速い。1つの巨大 PR は1週間以上放置されることが多い。
2. **リバートリスク分離** — critical バグ修正と secセキュリティ強化を同一 PR に混ぜると、片方で問題が出たときにまとめてリバートされる。
3. **blame 追跡** — 機能別の小さな commit にしておくと将来のデバッグが容易。

### issue と PR の使い分け

| 種別 | 使い分けの基準 |
|------|-------------|
| **PR** | 修正コードが明確で diff が小さい。バグ修正・ドキュメント誤記・設定ミス |
| **issue** | 修正に AMI 仕様確認や設計判断が必要。upstream (Isaac Lab) 側の変更を要するもの。コスト最適化の提案など |

具体的には:
- TERM バグ・dpkg ロック・AllowedCidr ドキュメント欠落 → **PR**（コードが明確）
- subnet-selector のアーキテクチャ変更 → **PR**（コードサンプルあり）
- `_isaac_sim` symlink / chown 問題 → **issue**（AMI 仕様確認が前提）
- EBS バックアップ・自動停止機能追加 → **issue**（設計判断が必要）

---

## 3. 第 1 PR ドラフト（即座にマージされそうなクイックウィン）

### 含める finding

| # | Finding | 重要度 |
|---|---------|--------|
| 1 | TERM 未設定で isaaclab.sh が常に失敗 | critical |
| 2 | unattended-upgrades との dpkg ロック競合 | high |
| 3 | README の `cd cdk` 誤記（ディレクトリ不存在） | high |
| 4 | AllowedCidr が README に記載なし | high |
| 5 | README パスワード設定コマンドに実インスタンス ID ハードコード | medium |
| 6 | Re-run UserData の誤パス `/var/lib/dcv-bootstrap/` | medium |

---

### PR タイトル

```
fix: critical userdata bugs (TERM, dpkg lock) and README deploy errors
```

### PR body (GitHub Markdown)

````markdown
## 概要

初回デプロイで必ず踏む critical/high バグと README の誤記を修正します。
g6e.2xlarge (ap-northeast-1) での実機デプロイで確認されたバグを含みます。

## 変更内容

### 1. `export TERM=xterm` を追加して Isaac Lab セットアップが常に失敗するバグを修正

**ファイル**: `isaacsim-workstation/lib/constructs/userdata_script.sh`

cloud-init / SSM 実行環境では `TERM` が未設定になる。Isaac Lab の `isaaclab.sh`
は先頭で `tabs 4` を実行するが、`TERM` 未設定の場合 `tabs` が exit 1 し
`set -e` によりスクリプト全体が終了する。その結果 `--conda` と `--install`
の両ステップが常に失敗し、Isaac Lab は一切セットアップされない。

> Note: `TERM=dumb` は ncurses が端末なしと判断するため tabs が失敗する。
> `TERM=xterm` (または vt100/linux) が必要。

```diff
- must "create-isaaclab-conda-env" '
-   ISAACLAB_DIR="/home/ubuntu/IsaacLab"
-   sudo -u ubuntu bash -c "
-     source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
-     if ! conda env list | grep -q env_isaaclab; then
-       cd $ISAACLAB_DIR && ./isaaclab.sh --conda
-     fi
-   "
- '
+ must "create-isaaclab-conda-env" '
+   ISAACLAB_DIR="/home/ubuntu/IsaacLab"
+   sudo -u ubuntu bash -c "
+     export TERM=xterm
+     source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
+     if ! conda env list | grep -q env_isaaclab; then
+       cd $ISAACLAB_DIR && ./isaaclab.sh --conda
+     fi
+   "
+ '

- must "fix-setuptools-and-install-isaac-lab" '
-   ISAACLAB_DIR="/home/ubuntu/IsaacLab"
-   sudo -u ubuntu bash -c "
-     source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
-     pip install setuptools==68.0.0
-     cd $ISAACLAB_DIR && ./isaaclab.sh --install
-   "
- '
+ must "fix-setuptools-and-install-isaac-lab" '
+   ISAACLAB_DIR="/home/ubuntu/IsaacLab"
+   sudo -u ubuntu bash -c "
+     export TERM=xterm
+     source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
+     pip install setuptools==68.0.0
+     cd $ISAACLAB_DIR && ./isaaclab.sh --install
+   "
+ '
```

### 2. unattended-upgrades との dpkg ロック競合を修正

**ファイル**: `isaacsim-workstation/lib/constructs/userdata_script.sh`

現在の `wait-for-dpkg-lock` には3つの問題がある。

- ロック解放の瞬間に break するが、unattended-upgrades がすぐに再取得する
- 60 回全てロック保持でタイムアウトした場合、最後の `sleep 10` の exit code 0 で
  `must()` が成功と判定し、後続 apt-get が dpkg lock で即失敗する
- apt-get のリトライが最大 48 秒（6 回 × 8 秒）と短すぎる

```diff
-must "wait-for-dpkg-lock" '
-  for i in $(seq 1 60); do
-    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
-      log "dpkg lock is free"; break
-    fi
-    log "Waiting for dpkg lock ($i/60)..."; sleep 10
-  done
-'
+must "wait-for-dpkg-lock" '
+  systemctl stop unattended-upgrades 2>/dev/null || true
+  for i in $(seq 1 60); do
+    if ! fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock >/dev/null 2>&1; then
+      log "dpkg lock is free"; break
+    fi
+    log "Waiting for dpkg lock ($i/60)..."; sleep 10
+    if [[ $i -eq 60 ]]; then
+      echo "ERROR: dpkg lock never released after 10 minutes" >&2; exit 1
+    fi
+  done
+'

 apt_install() {
   local pkgs="$*"
-  retry "apt-get install -yq --no-install-recommends $pkgs" "install: $pkgs" 6 8
+  retry "apt-get install -yq --no-install-recommends -o DPkg::Lock::Timeout=300 $pkgs" "install: $pkgs" 6 30
 }
```

### 3. README の Deploy セクションにある `cd cdk` を修正

**ファイル**: `isaacsim-workstation/README.md`, `isaacsim-workstation/README.ja.md`

`cdk` ディレクトリは存在しない。コピーペーストで実行した全ユーザーが最初のコマンドで失敗する。

```diff
-cd cdk
+cd isaacsim-workstation
```

### 4. 必須パラメータ `AllowedCidr` を README に追記

**ファイル**: `isaacsim-workstation/README.md`, `isaacsim-workstation/README.ja.md`

`AllowedCidr` は未指定だと synth 時に即エラーになる必須パラメータだが Configuration 表に記載がない。

```diff
 | Parameter | Description | Default |
 |-----------|-------------|---------|
 | StackPrefix | Prefix for resource names | DevWorkstation |
+| AllowedCidr | **[必須]** SSH/DCV/TensorBoard アクセスを許可する CIDR | なし |
 | VpcId | Existing VPC ID (creates new if empty) | (new VPC) |
```

デプロイコマンドの例も更新:

```diff
-cdk deploy
+cdk deploy -c AllowedCidr=$(curl -s checkip.amazonaws.com)/32
```

### 5. README のパスワード設定コマンドのハードコード値をプレースホルダーに変更

**ファイル**: `isaacsim-workstation/README.md`, `isaacsim-workstation/README.ja.md`

実インスタンス ID `i-0986d7fe5a672b6f5` と `us-east-1` がハードコードされている。

```diff
-aws ssm send-command --instance-id i-0986d7fe5a672b6f5 --region us-east-1 ...
+aws ssm send-command --instance-id <instance-id> --region <region> ...
```

### 6. Re-run UserData 手順の誤パスを修正

**ファイル**: `isaacsim-workstation/README.md`, `isaacsim-workstation/README.ja.md`

`/var/lib/dcv-bootstrap/` は存在しない。実際のパスは `/var/lib/workstation-bootstrap/`。

```diff
-sudo rm /var/lib/dcv-bootstrap/install-ros2-jazzy.done
+sudo rm /var/lib/workstation-bootstrap/install-ros2-jazzy.done
```

## テスト方法

- [ ] `cdk deploy -c AllowedCidr=<your-ip>/32` でスタック作成が成功することを確認
- [ ] インスタンス起動後 `sudo cat /var/log/workstation-bootstrap.summary` で全ステップが `STEP_OK` になることを確認（特に `create-isaaclab-conda-env` と `fix-setuptools-and-install-isaac-lab`）
- [ ] `conda activate env_isaaclab && python -c "import isaaclab"` が成功することを確認

## 関連 Issues

- Fixes: TERM 未設定による Isaac Lab セットアップ失敗
- Fixes: unattended-upgrades との dpkg ロック競合
````

---

## 4. 第 2-N PR / Issue リスト

優先度順に分類する。

### PR として投稿（修正コードが明確なもの）

#### PR-2: セキュリティ強化（IAM・IMDSv2・EBS暗号化）

**含める finding**:
- `AmazonEC2ContainerRegistryPowerUser` を `ReadOnly` に変更 (`isaacsim.ts` line 41)
- IMDSv2 強制（`httpTokens: 'required'`）の追加 (`isaacsim.ts` line 218)
- EBS ルートボリュームに `encrypted: true` を追加 (`isaacsim.ts` line 233)

```typescript
// isaacsim.ts での修正イメージ

// 1. ECR PowerUser → ReadOnly
ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly')

// 2. EBS 暗号化
volume: aws_ec2.BlockDeviceVolume.ebs(512, { deleteOnTermination: true, encrypted: true })

// 3. IMDSv2 強制 (escape hatch)
const cfnInstance = this.instance.node.defaultChild as aws_ec2.CfnInstance;
cfnInstance.metadataOptions = {
  httpTokens: 'required',
  httpPutResponseHopLimit: 1,
  httpEndpoint: 'enabled',
};
```

#### PR-3: cdk.context.json の漏洩対応

**含める finding**:
- `isaacsim-workstation/cdk.context.json` を `{}` に置き換え
- `.gitignore` に `isaacsim-workstation/cdk.context.json` を追加
- README に「cdk synth 後に生成される cdk.context.json はコミットしない」旨を追記
- git 履歴からの削除は別途対応（`git filter-repo` を推奨するが PR 範囲外）

#### PR-4: subnet-selector の過剰制約修正

**含める finding**:
- subnet-selector が全 30 インスタンスタイプを要求する問題
- 設定した InstanceType のみで AZ を検索するよう変更

```python
# index.py の修正イメージ
instance_type = event["ResourceProperties"].get("instanceType", "g6e.8xlarge")
offerings = ec2_client.describe_instance_type_offerings(
    LocationType="availability-zone",
    Filters=[{"Name": "instance-type", "Values": [instance_type]}],
)
az_candidates = sorted(
    [o["Location"] for o in offerings["InstanceTypeOfferings"]]
)
```

```typescript
// vpc.ts でも instanceType を渡す
properties: {
  selectedAZ: selectedAZ,
  instanceType: props.instanceType,
},
```

#### PR-5: S3 Files マウントの永続化（fstab 追記）

**含める finding**:
- インスタンス再起動後に `/mnt/s3files` が消える問題

```bash
# userdata_script.sh の mount-s3files ステップ末尾に追記
FS_ID="__S3FILES_FS_ID__"
FSTAB_LINE="${FS_ID} /mnt/s3files s3files _netdev,noresvport 0 0"
if ! grep -qF "$FS_ID" /etc/fstab; then
  echo "$FSTAB_LINE" >> /etc/fstab
fi
```

#### PR-6: `set -e` と `must()` の競合による SUMMARY 未出力の修正

**含める finding**:
- 最初の `must()` 失敗でスクリプトが終了し SUMMARY が出力されない問題

```diff
-set -Eeuo pipefail
+set -uo pipefail
```

#### PR-7: README の軽微な誤記まとめ修正

**含める finding**:
- `SimulatorAZNameParameterStore` の説明が「VPC Id」になっている誤記
- `environoment` typo (vpc.ts line 88)
- `package.json` の name フィールドタイポ (`issacsim` → `isaacsim`)
- `WaitForInstanceCommand` が UserData 完了を待たないことを README に明記

#### PR-8: AMI 名のワイルドカード化とバージョン固定解消

**含める finding**:
- `OV-Template-aws-ubuntu-isaac_sim-20260206T111303-prod-l4r5drddssotm` をワイルドカードに変更
- ami-lookup の `describe_images` に `Owners` フィルタ追加

#### PR-9: CFn カスタムリソースの PUT 失敗対応

**含める finding**:
- `send_cfn_response` が PUT 失敗を握り潰しスタックが 1 時間ハングする問題

```python
except Exception as e:
    logger.error(f"Error sending CFN response: {str(e)}")
    raise
```

---

### Issue として投稿（設計判断・調査が必要なもの）

| 優先度 | タイトル | 理由 |
|--------|---------|------|
| 高 | `_isaac_sim` symlink と `chown` 対象のパス不整合 | AMI 仕様確認が必要 |
| 高 | `chown -R ubuntu:ubuntu /opt/IsaacSim` の対象が広すぎる | AMI 仕様確認後に修正範囲を決定 |
| 中 | S3 Files マウントターゲットをプライベートサブネットに移動すべきか | アーキテクチャ設計判断 |
| 中 | VPC フローログの有効化 | コスト追加を伴う |
| 中 | インスタンス自動停止（EventBridge Scheduler）の追加 | 機能追加 |
| 中 | EBS バックアップ（AWS Backup）の追加 | 機能追加 |
| 中 | S3 Files バケットの `RemovalPolicy` を `RETAIN` に変更すべきか | DX とデータ保護のトレードオフ |
| 中 | README に Marketplace サブスクリプション確認手順を追加 | ドキュメント改善 |
| 低 | README に AdministratorAccess 推奨の削除と最小権限案内 | ドキュメント改善 |
| 低 | S3 バケットのアクセスログと明示的暗号化ポリシーの追加 | セキュリティ強化 |
| 低 | DCV 自己署名証明書についての説明追加（警告を無視させる案内の改善） | ドキュメント改善 |
| 低 | `isaacsim-workstation/README.ko.md` の追加（他サブプロジェクトとの一貫性） | 多言語対応 |

---

## 5. Issue テンプレート（TERM=unknown critical バグ）

以下をそのままコピーペーストして GitHub Issue として投稿できる。

---

**タイトル**: `[Bug] cloud-init / SSM 実行時に Isaac Lab セットアップが常に失敗する (TERM 未設定 + tabs 4)`

---

**本文**:

## バグ報告

### 概要

cloud-init / SSM 経由で実行される UserData スクリプトで、Isaac Lab の `--conda` および `--install` の両ステップが常に失敗します。その結果、Isaac Lab conda 環境が作成されず、インスタンス起動後に Isaac Lab を一切利用できない状態になります。

### 再現環境

- インスタンスタイプ: g6e.2xlarge（ap-northeast-1 で確認）
- デプロイ方法: `cdk deploy` → EC2 UserData 自動実行

### 症状

インスタンス起動後に以下を実行すると STEP_FAIL が記録されています。

```bash
sudo cat /var/log/workstation-bootstrap.summary
```

出力例:
```
STEP_FAIL: create-isaaclab-conda-env
STEP_FAIL: fix-setuptools-and-install-isaac-lab
```

`conda env list` を実行しても `env_isaaclab` が存在しません。

### 根本原因

cloud-init / SSM 実行環境では `TERM` 環境変数が未設定になります。Isaac Lab `release/2.2.0` の `isaaclab.sh` はスクリプト冒頭（`set -e` が有効な状態）で `tabs 4` を実行しますが、`TERM` が未設定の場合 `tabs` が exit code 1 で失敗し、`set -e` によりスクリプト全体が終了します。

```bash
# TERM 未設定での再現
env -i bash -c 'set -e; tabs 4'
# => bash: line 1: tabs: TERM environment variable not set.
# => exit code 1
```

なお `TERM=dumb` では解決しません。dumb 端末では ncurses がタブ設定コマンドを送出できないため同様に失敗します。

### 影響を受けるコード

`isaacsim-workstation/lib/constructs/userdata_script.sh`
- line 189: `./isaaclab.sh --conda`
- line 212: `./isaaclab.sh --install`

いずれも `sudo -u ubuntu bash -c "..."` 内で呼び出されており、親プロセスの `TERM` が継承されません。

### 修正方法

各 `sudo -u ubuntu bash -c` ブロックの先頭に `export TERM=xterm` を追加してください。

```bash
# create-isaaclab-conda-env ステップ
sudo -u ubuntu bash -c "
  export TERM=xterm          # ← この行を追加
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  if ! conda env list | grep -q env_isaaclab; then
    cd $ISAACLAB_DIR && ./isaaclab.sh --conda
  fi
"

# fix-setuptools-and-install-isaac-lab ステップ
sudo -u ubuntu bash -c "
  export TERM=xterm          # ← この行を追加
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
  pip install setuptools==68.0.0
  cd $ISAACLAB_DIR && ./isaaclab.sh --install
"
```

### 暫定ワークアラウンド

既デプロイのインスタンスでは、以下のコマンドで手動復旧できます。

```bash
# done マーカーを削除して再実行
sudo rm /var/lib/workstation-bootstrap/create-isaaclab-conda-env.done
sudo rm /var/lib/workstation-bootstrap/fix-setuptools-and-install-isaac-lab.done

# TERM を設定した状態でスクリプトを手動実行
export TERM=xterm
sudo -u ubuntu bash -c "
  export TERM=xterm
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  cd /home/ubuntu/IsaacLab && ./isaaclab.sh --conda
"
```

### 参考

- Isaac Lab release/2.2.0 の該当箇所: https://github.com/isaac-sim/IsaacLab/blob/release/2.2.0/isaaclab.sh
- ncurses `setupterm()` の TERM 未設定時の挙動: https://invisible-island.net/ncurses/man/curs_terminfo.3x.html

---

## 6. ボツにした finding（refuted）のリスト

adversarial verifier によって重要度を大幅に下方修正または「実害なし」と判定された 10 件。

### 1. EBS gp3 IOPS/スループット未設定（high → **medium に降格**）

**ボツ理由**: 「追加料金なしで最大 16000 IOPS / 1000 MB/s」という主要な impact 根拠が誤り。gp3 は 3000 IOPS / 125 MiB/s を超える分は追加課金が発生する（CDK の型定義にも明記）。コストフリーという前提が崩れるため severity を medium に降格。

---

### 2. S3FilesBucket に RemovalPolicy.DESTROY（critical → **medium に降格**）

**ボツ理由**: `cdk destroy` は明示的なユーザー操作が必要であり、偶発的・自動的なデータ削除ではない。README にも「全リソースが削除される」と明記されており、意図的な設計上のトレードオフ。サンプルキット用途では DESTROY が合理的な選択肢であり critical には該当しない。

---

### 3. インスタンス停止・アイドル自動停止なし（high → **medium に降格**）

**ボツ理由**: コードのバグではなく運用上のリスク（機能未実装）。README にコスト情報と手動停止の案内が既に記載されており、利用者に情報は提供されている。aws-samples のサンプルプロジェクトという文脈では high は過大評価。

---

### 4. CloudWatch アラーム・Budgets アラート未設定（high → **medium に降格**）

**ボツ理由**: サンプルキットかつデプロイ時に意図的なアクションが必要な構成のため、本番サービスと同列の high 評価は不適切。コストリスクは実在するが medium が適切。

---

### 5. DCV パスワード設定フローでのパスワード露出（high → **low に降格**）

**ボツ理由**: Finding の実際の内容はコマンド成功確認手順の欠如とドキュメント不備であり、主張していたパスワード平文露出リスクとは別物。README に `WaitForInstanceCommand` と SSM 代替手段が記載されており、影響はドキュメントの品質問題に限定される。

---

### 6. S3 Files マウントターゲットがパブリックサブネットに配置（medium → **low に降格**）

**ボツ理由**: EC2 インスタンス自体もパブリックサブネットに配置されており「設計意図との乖離」という主張は根拠が弱い。セキュリティグループで EC2 の SG からのみ port 2049 を許可しており、現状の実際の露出リスクは低い。「将来の SG 変更ミス」という仮定的シナリオへの依存度が高すぎる。

---

### 7. Lambda 関数が VPC 外でインターネット経由 AWS API アクセス（medium → **low に降格**）

**ボツ理由**: CloudFormation カスタムリソースの Lambda が VPC 外で実行されることは標準的な動作。また修正として「VPC 内プライベートサブネットへ配置」を提案しているが、このプロジェクトは `natGateways: 0` のため、プライベートサブネット内の Lambda が EC2 API に到達するには Interface VPC エンドポイントが別途必要であり、修正案が不完全。ファイル参照（`vpc.ts` vs `isaacsim.ts`）の誤りも含む。

---

### 8. `must()` 内サブシェルのエラー伝播問題（medium → **low に降格**）

**ボツ理由**: impact に誤りがある。「isaaclab.sh --install が失敗しても検出できない」と主張しているが、`./isaaclab.sh --install` は `bash -c` ブロックの最後のコマンドであるため、その失敗は exit code として正しく伝播し `must()` が確実に検出する。実際のリスクは前段の中間コマンド（conda activate 等）失敗のケースに限定される。

---

### 9. `workstation-stack.ts` で CfnOutput がアクセス不能（low で正しいが説明に誤り）

**ボツ理由**: タイトルの「Outputs が WorkstationStack レベルでアクセス不能」は誤り。CDK では construct 内部の `CfnOutput` はスタックレベルに自動的に合成される。「`noUnusedLocals` 警告」も `tsconfig.json` に `"noUnusedLocals": false` が明示されており発生しない。実在する問題は `const workstation` 変数が未使用というコードスタイルの軽微な問題のみ。

---

### 10. EBS バックアップ（AWS Backup）未設定（high → **medium に降格**）

**ボツ理由**: サンプルキットという性質上、本番 ML トレーニング環境を前提とした high 評価は不適切。S3 Files マウント経由でデータをバックアップできる仕組みが既に存在する（ユーザーが意識的に使う必要はあるが）。実害が生じるのはユーザーが EBS のみにデータを置き続けた場合に限られ、medium が適切。

---

*本報告書は 90 件の確認済み finding に基づき作成。6 つのレビューレンズ（security, reliability, ec2-platform-quirks, isaaclab-correctness, cost-and-ops, developer-experience）を横断的に評価した。*
