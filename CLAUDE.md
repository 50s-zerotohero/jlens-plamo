# CLAUDE.md — jlens-plamo プロジェクト指示書

## プロジェクト概要

`pfnet/plamo-3-nict-8b-base`(32層、GQA、Gemma型SWA/フルAttentionハイブリッド、GDN無し)に対して、
Anthropicの [Jacobian Lens (J-lens)](https://transformer-circuits.pub/2026/workspace/index.html) を
日本語で再現し、俳句の先読み計画(季語・切れ字・モーラ数)をJ-space越しに観察する。

**このリポジトリは研究プレビューであり、Anthropic論文の research-grade な再現ではない。**
WeZZard/jlens-qwen36 の免責文言に倣い、README冒頭に必ず明記すること:

> This is not evidence of consciousness. This is not proof that PLaMo has a Claude-like
> global workspace. This is not yet a robust reproduction of the Anthropic paper.

## 参照実装(必読・車輪の再発明禁止)

- **`anthropics/jacobian-lens`**(Apache-2.0) — J_ℓのフィッティング・適用ロジックの正本。
  `pip install git+https://github.com/anthropics/jacobian-lens` で依存として使う。
  `jlens.fitting` の数式(cotangentをターゲット位置で総和→ソース位置で平均)を再実装しない。
- **`WeZZard/jlens-qwen36`** — コーパスローダー設計・UI・プロジェクト構成の参考。
- **`neuronpedia/jacobian-lens`(HF Hub)** — オープンモデル向けフィッティングの config.yaml
  公開パターンを踏襲する(データセット名・split・シーケンス長・停止基準・実行コマンドを明記)。

## 確定済みの設計判断(Phase 1で決定済み・変更しない)

| 項目 | 決定内容 |
|---|---|
| モデル | `pfnet/plamo-3-nict-8b-base`(trust_remote_code=True) |
| 層数 | 32層、全層フィット(GDN無しなのでQwen3.5系のような自作Metal kernelは不要) |
| コーパス主成分 | `HuggingFaceFW/fineweb-2` の `jpn_Jpan` config(streaming） |
| コーパス補助 | 日本語Wikipedia(`wikimedia/wikipedia` ja config)10〜20%、長文一貫性の担保用 |
| 系列長 | 128トークン(論文の設定に合わせる) |
| 初回プロンプト数 | n=100(demo→research-gradeへの拡張余地を残す) |
| 俳句・韻文データ | **フィッティングコーパスに混ぜない**。Phase 5の held-out probe専用 |

---

## Phase 2 — データセット構築

### タスク
1. `data/corpus/build_corpus.py` を作成:
   - `datasets.load_dataset("HuggingFaceFW/fineweb-2", name="jpn_Jpan", split="train", streaming=True)`
     から候補を `take()` し、以下でフィルタ:
     - plamoのtokenizerで128トークン以上(切り詰めではなく、十分な長さの文書のみ採用)
     - 明らかなボイラープレート・ナビゲーション残骸を簡易ヒューリスティックで除外
     - 近似重複除去(MinHashまたは簡易n-gram Jaccard)
   - `wikimedia/wikipedia`(ja)から同様に補助分を抽出
   - 最終的に n=100(比率: fineweb-2 80〜85 / Wikipedia 15〜20)を `data/corpus/prompts.jsonl` に保存
     (各行: `{"text": ..., "source": "fineweb-2|wikipedia", "n_tokens": ..., "doc_id": ...}`)
2. `data/corpus/config.yaml` を作成し、Neuronpediaのconfig.yaml方式でレシピを明記:
   データセット名・config名・split・フィルタ条件・乱数シード・実行コマンド一式。
   **生テキストそのものは巨大なコーパスから抽出しているため、config.yamlとbuild_corpus.pyのみを
   コミットし、prompts.jsonlの再生成は誰でも同じ結果になるようseedを固定すること。**
3. 俳句用の held-out プローブセットを別途 `data/probes/haiku_prompts.jsonl` に用意
   (10〜20句程度、季語・切れ字・五七五構造を含む、Hideyaさんが手動でキュレーション予定 — Claude Codeは
   フォーマットの雛形とローダーのみ作成し、内容の選定はしない)

### 完了条件・チェックポイント
- `build_corpus.py` を実行して `prompts.jsonl` がちょうどn=100件生成されることを確認
- 生成された100件のうち5件をランダムに表示し、日本語として自然な文書か目視確認できるログを出す
- **ここで一度停止し、人間(Hideyaさん)にサンプルを提示して承認を得てからPhase 3へ進むこと**

---

## Phase 3 — J-lensフィッティング

### Phase 3a: スモークテスト(必須・先行)
本番フィッティングの前に、必ず以下を検証すること:

1. `jlens.from_hf()` がplamoの `trust_remote_code=True` モデルに対して
   `ActivationRecorder` のフックを正常に張れるか、2〜3プロンプトの極小フィットで確認
2. 得られる `J_ℓ` の shape が `[d_model, d_model]` として期待通りか
3. SWA層とフルAttention層の両方で勾配が正しく流れているか
   (SWA層で勾配が異常にゼロ・NaNになっていないか)

**このスモークテストが失敗する場合は、Phase 3bに進まず、まず互換性の問題を報告すること。**
(例: カスタムAttention実装がautograd互換でない、SWAのマスキングがVJPで壊れる等)

### Phase 3b: 本番フィッティング
1. `scripts/run_fit.py` を作成。要件:
   - `jlens.fit(model, prompts, checkpoint_path=...)` を32層全層に対して実行
   - **チェックポイント・resume機能を必ず実装**(`JacobianLens.merge()` を使い、
     プロンプトのdisjointなスライスごとに個別フィット→マージできる構成にする。
     数時間かかるジョブが途中で落ちても再開できるようにする)
   - 進捗ログ(現在の層・プロンプト・経過時間・推定残り時間)を標準出力に出す
2. 実行は人間が手動で行う想定(バックグラウンドで数時間走らせる)。
   Claude Codeはスクリプトの作成とテストまでを担当し、**本番フィット自体の実行完了を待って
   ブロックしない**こと(長時間ジョブはユーザーに委ねる)。

### 完了条件・チェックポイント
- フィット済みレンズ(`.npz`または`.pt`)が生成される
- `data/lens/README.md` に、Neuronpediaのconfig.yaml同様、使用コーパス・件数・層数・
  フィッティング時間・実行コマンドを記録する
- 簡易な定性チェック用スクリプト(下記Phase 4の一部を先行実行)で、日本語の二段推論プロンプト
  (例: 「県庁所在地が松山である県は」→中間層で「愛媛」、終盤層で答えが出るか)を試し、
  読み出しが意味を持つ語になっているか確認する
- **ここで一度停止し、読み出し結果のサンプルを人間に提示して品質を確認してもらうこと。
  ノイズだらけであれば、コーパスをn=100→300程度に拡張して再フィットする判断をここで行う**

---

## Phase 4 — 推論コード

### タスク
1. `jlens_plamo/apply.py`: `jlens.JacobianLens.apply()` をラップし、
   プロンプト・位置・層を指定すると各層のtop-k読み出しトークンを返すAPI/CLIを作成
2. `jlens_plamo/interventions.py`(任意・stretch): steer/swap/ablateの基本実装
   (WeZZard実装の`interventions.py`を参考にするが、コード自体は独自実装または
   `anthropics/jacobian-lens`の該当部を利用)
3. 定性デモ: 日本語の二段推論例、多言語例(英↔日切り替え)などで
   `spider→legs` に相当する日本語版の事例を最低2〜3個再現する

### 完了条件・チェックポイント
- CLIから任意のプロンプトを渡して layer × position のtop-1トークン表が出力できる
- デモ例のスクリーンショット/出力ログをREADMEに掲載できる状態にする

---

## Phase 5 — 俳句拡張 + UI

### タスク
1. **俳句プロービング実験**(オリジナル拡張・本プロジェクトの新規性の核):
   `data/probes/haiku_prompts.jsonl` の句を、Phase 3で作った汎用フィット済みレンズに通し、
   初句(5)の時点で中間層に季語・季節・切れ字に関連する語が先読みで浮かぶかを観察する
   - 介入実験(swap)も可能であれば: ある季語の概念を別の季語に入れ替えて、
     後続の生成が変化するか確認(論文のpoetry-rhyme実験と同型)
2. UI: WeZZard実装の `web/index.html`(position × layer のスライスグリッド、
   クリックでtop-10表示)を参考に、日本語トークン(マルチバイト・サブワード分割)の
   表示崩れに注意して自己完結HTMLビューアを作成
   - バックエンドはFastAPIで `/api/lens`, `/api/slice`, `/generate` 程度の最小構成

### 完了条件・チェックポイント
- ローカルでUIサーバーが起動し、俳句プロンプトを入力してスライスグリッドが表示される
- 俳句先読みの事例が最低1つ、スクリーンショット付きでドキュメント化されている
- 発見が「解釈可能だがノイジー」なのか「明確な先読みパターン」なのか、
  正直に記述する(誇張しない。WeZZard実装の "hypothesis-generating rather than
  a robust reproduction" のトーンを踏襲する)

---

## GitHub公開に向けた横断要件

1. **ライセンス**
   - コード: Apache-2.0(`anthropics/jacobian-lens`と揃える)
   - README内に、PLaMoモデル自体は別ライセンス(PLaMo Community License)であることを明記。
     具体的には: "Built with PLaMo" の表示義務、モデル名に"PLaMo"を含める要件(該当する場合)、
     商用利用時の条件(年間売上10億円以下等)をREADMEにリンクとともに記載する
   - 生コーパス(fineweb-2, Wikipediaの実データ)はリポジトリにコミットしない。
     `config.yaml` + 生成スクリプトのみを公開する(`anthropics/jacobian-lens`の
     "No model weights or text corpora are bundled" 方針に倣う)
2. **README構成**(WeZZard実装のREADME構成を踏襲):
   - 免責文言(冒頭)
   - Status(demo-quality / research-grade のどちらか、正直に)
   - Quick start
   - How it works(J_ℓの数式、コーパスの選定理由)
   - Project layout
   - Limitations(必ず記載: n=100の限界、plamo固有の未検証事項、俳句実験の解釈上の注意)
   - Acknowledgements(anthropics/jacobian-lens, WeZZard/jlens-qwen36 への謝辞)
3. 各PhaseのPRは分割してコミットする(Phase単位でsquashしない)。
   後から「どのコーパス・どのフィットでこの結果が出たか」を追跡できるようにするため。

---

## Claude Codeへの運用上の指示

- 各Phaseの「完了条件・チェックポイント」に達したら、**そこで作業を止めて人間に報告し、
  次のPhaseへの着手許可を待つこと**。Phase 2→5を無停止で自動的に流さない。
- Phase 3b(本番フィッティング)は長時間のGPUジョブなので、スクリプト作成・スモークテストまでを
  Claude Codeが担当し、実行そのものはユーザーが手動でキックする前提とする。
- 不明点(特にplamoのcustom_codeとの互換性、SWAのフック挙動)は、憶測で実装を進めず、
  小さい検証コードで先に事実を確認してから本実装に進むこと。
