# 顧客コール準備 実行計画

**Total runtime**: ~175 分  /  **Cost**: ~$16

**Concurrency**: 2インスタンス並行実行。Instance A (g6e.4xlarge i-0c27103cc9dc53e0f) では order 3→4→5→7 を直列実行。Instance B (g6e.2xlarge i-03bfb5f08ce61d4b5) では order 2→6 を直列実行。order 1（ローカル dmon_stats.py 編集）は両インスタンスの起動と同時に実行する。クリティカルパスは Instance A の order 3→4→5→7 (155 分)。Instance B の order 2→6 (95 分) はクリティカルパスより短いため待ち時間なし。

**Critical path**: order 1（ローカル 20 分）と並行して order 3 を開始 → order 3（contact scope, 30 分）→ order 4（solver iter, 45 分）→ order 5（OOM sweep, 40 分）→ order 7（combined stack, 40 分）。合計 Instance A ウォールタイム 155 分。order 1 の 20 分はオーバーラップ。

---

## 実行順

### Step 1: dmon_stats.py に fb(MB) 列を追加し既存全ログを再解析する

- **Instance**: ローカル (EC2 不要)
- **Profile / script**: /Users/akazawt/work/zenmai-lab/experiments/isaac-lab-bench/scripts/dmon_stats.py を直接編集。Stats dataclass に fb_avg/fb_max を追加し parse_file() で col 14 を読む。emit_text/csv/md に列を追加後、既存3本の dmon ログに対して python3 scripts/dmon_stats.py --format md results/*/run-seed*/nvidia-smi-dmon.log を実行する。
- **Why this order**: コスト $0、所要 20 分、EC2 不要のローカル作業。「num_envs=16384 で VRAM 33% (15.2 GB) しか使っていない」という数値を公式ツールとして出力可能にする。NOTES.md L160 の open question を解決し、推奨「num_envs を上げろ」の安全根拠を確定させる。他の全ステップと並行してすぐに始められるため order=1 とする。同時に Instance B での Flat ベースライン計測（order 2）を開始する。
- **Success criteria**: dmon_stats.py --format md の出力に fb_avg_mb / fb_max_mb 列が追加され、n4096: max_fb≈7,297 MB / n8192: ≈9,951 MB / n16384: ≈15,237 MB の3点が一致すること。

### Step 2: G1-Flat ベースライン計測 (Instance B 並行起動)

- **Instance**: g6e.2xlarge (i-03bfb5f08ce61d4b5)
- **Profile / script**: orchestrate.sh に g1-flat プロファイルを追加し Isaac-Velocity-Flat-G1-v0 タスクを実行。SEEDS='42 123 456' ITERS=100 NUM_ENVS=4096。事前に benchmark_rsl_rl.py --list-tasks でタスク存在確認。order 1 の dmon_stats.py 修正と並行して開始する。
- **Why this order**: order 1 と同時に開始できる独立した実験。Rough+4096 ベースライン 71k fps に対し Flat での fps 上昇量（height_scan 除去によるコスト削減）を定量化する。order 6（ABL-3 heightscan比較）および order 7（ABL-6 combined stack）で引用する「Flat 基準値」を確定させるために order 2-3 の間に完了している必要がある。
- **Success criteria**: summary.csv に mean_total_fps が記録され NA でないこと。Rough ベースライン 71k fps より有意に高い fps（予想 90k〜120k fps）が記録されること。

### Step 3: PhysX-ABL-2: ContactSensor prim_path を全23リンク→足首2リンクのみに絞る

- **Instance**: g6e.4xlarge (i-0c27103cc9dc53e0f)
- **Profile / script**: rough_env_cfg.py __post_init__ に ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*_ankle_roll_link', history_length=3, track_air_time=True) を追加パッチ。SEEDS='42 123 456' ITERS=100 NUM_ENVS=16384 bash orchestrate.sh g1。タグ: physx-abl2-contact-scope。ベースライン比較対象: sweep-summary.csv の 16384 行 (112,503 fps, MEM max 96%)。
- **Why this order**: must_run_before_call=true (ROI 8)。16k env × 23 リンク → 16k × 2 リンクという 11.5× の削減比率はメモリ帯域への直接的な影響が期待でき、MEM 飽和状態（96%）からの delta が最も顕著に出るセットアップ。「ContactSensor のスコープを実際に使うリンクのみに絞れ」という即効性の高い推奨を数値付きで伝えるために必須。order 4 (solver iter) と同じ Instance A で直列実行（order 3 完了後に order 4 開始）。
- **Success criteria**: MEM max が 96% から有意に（2pp 以上）低下するか、fps が 112k より有意に上昇すること。あるいは変化なしという陰性結果でも「帯域原因は contact sensor ではない」の証拠として記録する。

### Step 4: PhysX-ABL-1: ソルバーイテレーション削減 (pos 8→4, vel 4→1)

- **Instance**: g6e.4xlarge (i-0c27103cc9dc53e0f)
- **Profile / script**: rough_env_cfg.py に ArticulationRootPropertiesCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=1) パッチ。order 3 のパッチを revert してから適用。SEEDS='42 123 456' ITERS=100 NUM_ENVS=16384。タグ: physx-abl1-solver-iter。
- **Why this order**: ROI 7。order 3 と同じ Instance A で直列実行。ContactSensor 実験（order 3）の結果を確認してから実施することでボトルネックの移動（MEM が下がれば SM が次のボトルネックになる）を連続して観察できる。ソルバーイテレーション削減単体の fps delta を order 7（combined stack）に供給する。
- **Success criteria**: fps が +10% 以上改善する（デフォルト設定の保守性を示す）、または差が出ない（ソルバーコストは帯域飽和の原因でないことを示す）こと。max_rewards の急激な低下（>15%）がないこと。

### Step 5: num_envs OOM 閾値の特定（24576 / 32768）

- **Instance**: g6e.4xlarge (i-0c27103cc9dc53e0f)
- **Profile / script**: order 4 のパッチを revert してデフォルト設定に戻す。NUM_ENVS_SWEEP='24576 32768' SEEDS='42' ITERS=50 bash orchestrate.sh g1-num-envs。OOM 時は bench.log に NA が記録される。dmesg | grep -i oom も確認。
- **Why this order**: must_run_before_call=true 相当（ROI 8）。「安全な最大 num_envs = N」という数値は顧客への「num_envs を上げろ」推奨に必須の安全根拠。order 3/4 のアブレーション後にデフォルト設定で計測するため、ここで自然に order 3/4 のパッチを全て元に戻す。OOM 境界が確定すれば order 1 の VRAM テーブルと合わせて「n4096=7.3 GB / n16384=15.2 GB / 上限 N 確定」という完全な推奨が完成する。
- **Success criteria**: 24576 または 32768 のいずれかで OOM または fps 頭打ちが確認され、安全上限 N が確定すること。あるいは両者とも動作し「46 GB VRAM でも 32k envs まで余裕あり」が判明すること。

### Step 6: PhysX-ABL-3: height scan 3点比較 (None / update_period 2× / resolution削減)

- **Instance**: g6e.2xlarge (i-03bfb5f08ce61d4b5)
- **Profile / script**: order 2 の Flat ベースライン完了後に同 Instance B で実施。パターン A（height_scanner=None, terrain=plane）、パターン B（update_period 2×）、パターン C（resolution=0.15, size=[1.0,0.6]）の3条件を直列実行。各パターン SEEDS='42 123' ITERS=100 NUM_ENVS=16384。タグ: physx-abl3-heightscan-{none,halffreq,lowres}。
- **Why this order**: ROI 7。order 2 の Flat ベースラインが完了した Instance B で引き続き実施（インスタンス効率化）。ABL-3 は ABL-2/ABL-1 と独立しているため Instance A の order 3-5 と並行運転可能。3パターンの fps 序列（A > B > C）が確認できれば raycast コストの定量的な証明として order 7 の combined stack 設計に直接使える。
- **Success criteria**: パターン A（height scan 完全削除）が最大 fps 上昇（予想 +10〜25%）を示し、B < C の序列が確認されること。3条件すべてで NA でない fps が記録されること。

### Step 7: PhysX-ABL-6: ABL-1〜3 ベスト設定の組み合わせ「最適化スタック」ベンチ

- **Instance**: g6e.4xlarge (i-0c27103cc9dc53e0f)
- **Profile / script**: order 3〜6 の結果を見て効果の大きかった項目（例: solver iter 削減 + contact scope 絞り込み + height scan halffreq または none）を組み合わせた config を作成。SEEDS='42 123 456' ITERS=100 NUM_ENVS=16384。タグ: physx-abl6-combined-stack。dmon_stats.py で SM/MEM/PWR 全体プロファイルを取得。order 3, 4, 6 の完了を待ってから実施。
- **Why this order**: must_run_before_call=true（ROI 7）。「最適化後の EC2 何倍速」という顧客ピッチの核心数値を確定させる唯一の実験。order 3〜6 の個別結果が揃ってから初めて「組み合わせると加算的か、干渉するか」を実測できる。order 5（OOM sweep）が Instance A で完了した後に最後のスロットで実行する。
- **Success criteria**: 組み合わせ後の fps が個別 ablation の単純加算予測の 80% 以上を達成すること。あるいは干渉が確認された場合はその理由が SM/MEM バランスから説明できること。summary.csv に NA がないこと。

---

## オプション (時間が余ったら)

- PhysX-ABL-4 (use_fabric=True 確認): ROI 4、confirms-known のためスキップ推奨。顧客に口頭で「use_fabric=True を確認してください」と伝えれば十分。
- PhysX-ABL-5 (GPU バッファ overflow 診断): 既存ログ grep 済みで foundLostPairs エラーゼロ確認済み。スキップ。
- PhysX-ABL-7 (enhanced_determinism fps コスト): ROI 4、confirms-known。コールで顧客に直接確認する方が速い。
- PhysX-ABL-8 (contact history_length 3→1): ROI 3、帯域への影響 0.4% 未満と文書化済み。スキップ。
- g6e.8xlarge PCIe x16 確認: ROI 5、PCIe はボトルネックでないことが SM/MEM データから既に主張可能。スキップ。
- g7e.2xlarge Blackwell GDDR7 計測: ROI 6、ドライバ互換リスク medium + 120 分。コール後のフォローアップ実験として推奨。
- g6e.12xlarge 4-GPU DDP: ROI 5、--distributed フラグの動作確認未了 + 105 分のリスク。コール後。
- decimation 4 vs 8: ROI 6、顧客の実際の decimation 設定を先に確認してから実施判断。
- Isaac Lab v2.2 vs v2.3 FPS 差分: ROI 3、90 分 + medium リスク。コールで顧客バージョンを先に確認。
- G1-Flat + 16384 envs VRAM 上限: ROI 5、order 2 の Flat ベースライン後に時間があれば追加。
- 顧客向け診断スクリプト群 (diagnose_gpu.sh, check_config.py): ROI 4〜5。コール後にフォローアップ資料として整備。
- 5シード FPS ばらつき (n=16384): ROI 6。order 7 完了後に Instance A が空いていれば追加実行可。

---

## 成果物

- `/Users/akazawt/work/zenmai-lab/experiments/isaac-lab-bench/scripts/dmon_stats.py`
  - fb_avg_mb / fb_max_mb 列を追加した更新版。既存3本のdmonログから VRAM 実使用量テーブル（n4096=7.3 GB / n8192=9.9 GB / n16384=15.2 GB）を出力する。顧客への「num_envs を上げても VRAM は余裕あり」推奨の定量根拠として提示する。
- `/Users/akazawt/work/zenmai-lab/experiments/isaac-lab-bench/results/physx-abl2-contact-scope-*/summary.csv`
  - ContactSensor prim_path 絞り込み実験（全23リンク→足首2リンク）の fps / MEM% 比較結果。「設定1行で MEM 帯域を X% 削減できる」という顧客向け即効性推奨の実測値。
- `/Users/akazawt/work/zenmai-lab/experiments/isaac-lab-bench/results/physx-abl1-solver-iter-*/summary.csv`
  - G1 ソルバーイテレーション削減（pos 8→4, vel 4→1）の fps / max_rewards 比較。「デフォルト設定は AnymalC 実績値の2倍に設定されている」仮説の実測による確認または棄却。
- `/Users/akazawt/work/zenmai-lab/experiments/isaac-lab-bench/results/physx-abl3-heightscan-*/summary.csv`
  - height scan 3パターン比較（None / halffreq / lowres）の fps 序列データ。「raycast コストが帯域ボトルネックの主要因」という主張の定量的証明または反証。
- `/Users/akazawt/work/zenmai-lab/experiments/isaac-lab-bench/results/g1-flat-*/summary.csv`
  - G1-Flat（height_scanner=None）のベースライン fps。Rough+4096(71k fps) との比較で「Flat 環境は Rough 比 X倍速い」を実数で示す。顧客が rough 地形不要なら即日適用できる最大スループット改善案の根拠。
- `/Users/akazawt/work/zenmai-lab/experiments/isaac-lab-bench/results/g1-oom-sweep-*/summary.csv`
  - num_envs=24576/32768 での OOM 境界確定データ。「g6e.4xlarge での安全な最大 num_envs = N」を具体的な数値で顧客に提示するための根拠。VRAM テーブルと組み合わせて1枚のスライドに整理できる。
- `/Users/akazawt/work/zenmai-lab/experiments/isaac-lab-bench/results/physx-abl6-combined-stack-*/summary.csv`
  - ABL-1〜3 のベスト設定を組み合わせた「最適化スタック」の fps 実測値。「最適化後の EC2 G1 Rough は X fps = 顧客環境より Y倍速い」という顧客ピッチの核心数値を含む。30h→20h 短縮目標の達成可否を確定させる。

---

## Meeting Playbook

## カスタマーコール進行プレイブック

### 開始〜5分: アイスブレイクと環境ヒアリング（必須質問リスト）

顧客が参加したら最初の5分で以下を確認する。これは後続の提案内容を顧客環境に合わせるために必須。

**質問1（最重要）:** 「TensorBoard の Perf/collection_time と Perf/learning_time の比率を確認できますか？もしくは学習ログに "Collection time:" という行があれば数値を教えてください。」
→ 回答が collection:learn = 10:1 以上なら「環境側最適化が効く」と断言。3:1 以下なら PPO 側のチューニングが先。

**質問2:** 「現在の num_envs と、nvidia-smi --query-gpu=memory.used --format=csv の出力値を教えてください。」
→ 4096 以下なら「まず 8192 に上げるだけで 1.36× fps 改善が見込めます」と即提案。

**質問3:** 「Isaac Lab のバージョンは？学習スクリプトは公式の train.py ですか、カスタムですか？」
→ カスタムなら TF32 フラグ確認を促す（PERF-LEVERS §1.4）。

**質問4:** 「ContactSensorCfg の prim_path は "{ENV_REGEX_NS}/Robot/.*"（全リンク）のままですか？」
→ Yes なら order 3 の実測結果（ABL-2）を提示して「足首2リンクに絞るだけで MEM 帯域を削減できます」と具体的に説明。

**質問5:** 「rough 地形での汎化は学習要件ですか？それとも最初はフラット地形で訓練できますか？」
→ Flat 可能なら order 2 の結果（Flat fps vs Rough fps）を提示して最大スループット向上を提案。

---

### 5〜20分: 現状分析の提示（既存データ）

NOTES.md 実験3の結果を使って以下を順番に見せる。

**チャート1: num_envs vs fps スケーリング曲線**
n4096=72k / n8192=99k / n16384=113k の3点グラフ。
メッセージ: 「デフォルト 4096 環境では GPU が SM 48% しか使われていません。16384 に上げると 1.56× になります。」

**チャート2: VRAM 使用量テーブル（order 1 の成果）**
n4096=7.3 GB (16%) / n8192=9.9 GB (22%) / n16384=15.2 GB (33%) / 46 GB L40S。
メッセージ: 「VRAM は 16384 環境でも 33% しか使っていません。OOM を心配せずに num_envs を増やせます。」
→ order 5 の結果が出ていれば「安全上限は N 環境と実測で確認済み」と追加。

**チャート3: GPU 使用率サマリ**
n16384: SM max 93% / MEM max 96% → 「メモリ帯域律速が確認済み」。
メッセージ: 「L40S の GDDR6 帯域 864 GB/s が本当のボトルネックです。CPU でもストレージでもありません。」

---

### 20〜35分: 最適化推奨の提示（実験結果付き）

**推奨A（即日適用）: num_envs を 8192 以上に上げる**
コード変更なし、1行変更で 1.36× fps。「まずこれをやってください」。

**推奨B（ContactSensor prim_path 絞り込み）: order 3 実験結果を提示**
変更前の全23リンク設定が MEM 帯域を X% 消費していたことを実測値で示す。
コード例（PERF-LEVERS §2.1）をその場で貼り付け可能な状態で準備。

**推奨C（height scan コスト削減）: order 6 実験結果を提示**
パターン A/B/C の fps 序列を表で示す。
「rough 地形不要なら height_scanner=None で最大 Y% 向上」（Flat ベースライン数値を引用）。

**推奨D（ソルバーイテレーション削減）: order 4 実験結果を提示**
AnymalC 実績値（pos=4, vel=0）との比較で「デフォルトは保守的すぎる」または「差が出ない＝他に律速あり」を実測結果に基づいて断言。

**推奨E（組み合わせ最適化スタック）: order 7 実験結果を提示**
「これらを組み合わせると X fps になります = 顧客の RTX 5090 環境より Y 倍速い」。
30h → Z h への短縮見込みを計算して提示。

---

### 35〜45分: 中期インスタンス推奨と Q&A

**インスタンス推奨**
現在の最適化でも不足なら:
- g6e.4xlarge → num_envs=16384 で 1.56× fps（$3.25/hr 相当）
- g7e.2xlarge（Blackwell GDDR7 1597 GB/s, 理論帯域 1.85× L40S）→「計測準備中、コール後に数値を提供します」と伝える

**顧客固有の誤設定診断（PERF-LEVERS §1.3）**
python3 -c で use_fabric / enable_enhanced_determinism / enable_ccd を今すぐ確認するワンライナーを口頭で案内。
「実行結果をチャットに貼ってもらえますか？」

**クロージング質問**
「today のアクションとして何から始めますか？num_envs を上げることから始められますか？」
次回フォローアップの日程と、コール後に送る資料（ContactSensor コード例、dmon 診断コマンド）をその場で約束する。