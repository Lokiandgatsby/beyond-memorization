"""Build the technical report from stored aggregate results."""
import argparse
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage


def create(root):
    root=Path(root);docs=root/"docs";docs.mkdir(exist_ok=True)
    agg=json.loads((root/"results/aggregated.json").read_text())
    stats=json.loads((root/"results/study_statistics.json").read_text())
    lookup={(r["model"],r["condition"],r["split"]):r for r in agg}
    get=lambda m,c,s:lookup[m,c,s]
    fmt=lambda r:f'{r["test_mean"]*100:.2f} ± {r["test_sd"]*100:.2f}'
    required_fonts=["DejaVuSerif.ttf","DejaVuSans.ttf","DejaVuSans-Bold.ttf"]
    configured=os.environ.get("BEYOND_MEMORIZATION_FONT_DIR")
    candidates=[Path(configured)] if configured else []
    candidates += [Path("/usr/share/fonts/truetype/dejavu"),Path("/usr/local/share/fonts"),Path("/Library/Fonts")]
    font_dir=next((directory for directory in candidates if all((directory/f).is_file() for f in required_fonts)),None)
    if font_dir is None:
        raise FileNotFoundError("Install DejaVu fonts or set BEYOND_MEMORIZATION_FONT_DIR to the font directory")
    for font,file in zip(["Body","Sans","Bold"],required_fonts):
        pdfmetrics.registerFont(TTFont(font,str(font_dir/file)))
    styles={
      "title":ParagraphStyle("title",fontName="Bold",fontSize=27,leading=33,textColor=colors.HexColor("#0f172a"),spaceAfter=12),
      "subtitle":ParagraphStyle("subtitle",fontName="Sans",fontSize=12,leading=18,textColor=colors.HexColor("#334155"),spaceAfter=14),
      "h":ParagraphStyle("h",fontName="Bold",fontSize=13,leading=18,textColor=colors.HexColor("#0f766e"),spaceBefore=12,spaceAfter=7),
      "body":ParagraphStyle("body",fontName="Body",fontSize=9.5,leading=14.5,spaceAfter=9),
      "small":ParagraphStyle("small",fontName="Sans",fontSize=8,leading=11.5,spaceAfter=6,textColor=colors.HexColor("#475569")),
      "cell":ParagraphStyle("cell",fontName="Sans",fontSize=8.2,leading=12),
    }
    story=[];md=[]
    def p(text,style="body"):
        story.append(Paragraph(escape(text).replace("\n","<br/>"),styles[style]));md.append(text+"\n")
    def h(text):
        p(text,"h");md[-1]="## "+text+"\n"
    def page():story.append(PageBreak());md.append("\n---\n")
    def table(rows,widths):
        data=[[Paragraph(escape(str(c)),styles["cell"]) for c in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e2e8f0")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LINEBELOW",(0,0),(-1,0),.8,colors.HexColor("#94a3b8")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafc")])]))
        story.extend([t,Spacer(1,8)])
        for i,row in enumerate(rows):
            md.append("| "+" | ".join(map(str,row))+" |\n")
            if i==0:md.append("| "+" | ".join(["---"]*len(row))+" |\n")
    def figure(name,width=500,maxheight=300):
        file=root/"figures"/name
        with PILImage.open(file) as im:w,hi=im.size
        scale=min(width/w,maxheight/hi);story.append(Image(str(file),width=w*scale,height=hi*scale));story.append(Spacer(1,5))
        md.append(f"![{name}](../figures/{name})\n")
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"));canvas.line(44,43,A4[0]-44,43)
        canvas.setFont("Sans",7);canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(44,30,"BEYOND MEMORIZATION  •  EXPERIMENT REPORT")
        canvas.drawRightString(A4[0]-44,30,str(doc.page))
    def build(path,items):
        SimpleDocTemplate(str(path),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=57,
                          title="Beyond Memorization",author="").build(items,onFirstPage=footer,onLaterPages=footer)

    # Page 1.
    p("Beyond Memorization","title")
    p("A controlled study of symmetry-aware evaluation and generalization in small neural networks","subtitle")
    p("Modular addition • Symmetry-aware evaluation • Experiment report","small")
    h("Abstract")
    p(f"We study modular addition over 97 classes using exact-count ordered-pair and swap-orbit-grouped partitions. An embedding MLP and a one-block causal Transformer are evaluated over three seeds. A fixed matrix contains 24 runs at 3,000 optimizer updates; six explicitly exploratory Transformer runs extend the horizon to 12,000. At the base setting, MLP test accuracy is {fmt(get('mlp','main','ordered'))}% on ordered partitions and {fmt(get('mlp','main','grouped'))}% on grouped partitions (mean ± sample SD). A training-label lookup using operand reversal obtains {get('mlp','main','ordered')['lookup_mean']*100:.2f}% on ordered test partitions and {get('mlp','main','grouped')['lookup_mean']*100:.2f}% on grouped partitions without implementing arithmetic. Transformer results depend strongly on the joint training regime and evaluation protocol. No run meets our coarse delayed-generalization criterion. Configurations, learning curves, final checkpoints and per-example test predictions are retained for reproducibility.")
    h("1  Introduction")
    p("A model may predict unseen ordered inputs successfully even when those inputs belong to mathematical equivalence classes already present in training. For addition, (a,b) and (b,a) share a target. Holding out an ordered pair therefore differs from holding out its whole swap orbit. This project asks how that evaluation choice interacts with architecture, data fraction and regularization. Exact input duplicates are excluded in both protocols. Cross-partition reversal is not called an implementation leak: it is a property of the ordinary ordered-input task.")
    h("2  Related work and scope")
    p("Power et al. [1] studied delayed generalization on small algorithmic datasets. Nanda et al. [2] analyzed modular-addition Transformers and used Fourier-space interventions to study learned mechanisms. Gromov [3] examined a simpler neural-network setting for modular arithmetic. These works motivate a tractable task; our models and optimizer schedule are not exact reproductions. We investigate evaluation and finite-budget behavior rather than reverse-engineering a circuit. The comparison focuses on orbit-based evaluation and learning dynamics.")

    # Page 2.
    page();h("3  Method")
    p("Dataset. For all a,b in {0,...,96}, the target is y=(a+b) mod 97. Inputs contain only integer tokens. There are 9,409 ordered pairs and 97 target classes. The arithmetic formula generates labels; it is never supplied to the learned predictor. No external dataset, download or annotation is required.")
    table([["Training fraction","Train","Validation","Test"],["0.60 (base)",5645,1881,1883],["0.30 (ablation)",2822,1881,4706]],[180,100,110,110])
    p("Partitions. Ordered splits use a seeded permutation of all pairs. Grouped splits allocate unordered two-element swap orbits and singleton diagonal orbits. A parity-constrained allocation gives exactly the same train/validation/test counts as the ordered scheme; diagonal counts are chosen near proportional allocation. Every pair appears once, and no swap orbit crosses a grouped boundary. Class histograms, diagonal counts and byte hashes are retained for auditing. The algorithms do not force the label histograms or orbit coverage to match.")
    main=json.loads((root/"results/mlp_main_ordered_s0/summary.json").read_text())
    trans=json.loads((root/"results/transformer_main_ordered_s0/summary.json").read_text())
    table([["Model","Architecture","Parameters"],
           ["MLP","Shared 32-d token embedding; concatenation; two 128-unit GELU hidden layers",f'{main["parameters"]:,}'],
           ["Transformer","[a,b,=] tokens; learned positions; one causal pre-norm block, 4 heads, width 32; 128-unit GELU feedforward",f'{trans["parameters"]:,}']],[90,335,75])
    h("3.1  Optimization and controlled matrix")
    p("All runs use AdamW, learning rate 0.003, betas (0.9,0.98), no dropout or schedule, and 256 training pairs drawn with replacement per update. Weight decay applies to all parameters. Base weight decay is 1.0. Each architecture runs both splits at training fraction 0.6; Transformer alone also runs fraction 0.3 and decay 0, one factor at a time. Each setting uses seeds 0,1,2. The same seed fixes initialization, partition generation and a separate batch-sampling generator. All recorded runs used one PyTorch CPU thread per worker.")
    h("3.2  Evaluation and dynamics")
    p("Training and validation are measured every 100 updates, including update 0. The final fixed-step checkpoint is evaluated on test; validation does not select a checkpoint. No test curve is used for optimization. Error strata and swap consistency are post-training diagnostics. We flag a delayed-generalization proxy only when first logged train accuracy ≥99% precedes first validation accuracy ≥90% by at least 500 updates. This is a coarse criterion, not proof of an abrupt transition. Missing crossings are censored at the run horizon.")

    # Page 3.
    page();h("4  Fixed-horizon results: 3,000 updates")
    rows=[["Model / condition","Ordered test (%)","Grouped test (%)"]]
    labels=[("mlp","main","MLP / 60%, decay 1"),("transformer","main","Transformer / 60%, decay 1"),
            ("transformer","fraction","Transformer / 30%, decay 1"),("transformer","decay","Transformer / 60%, decay 0")]
    for m,c,label in labels:rows.append([label,fmt(get(m,c,"ordered")),fmt(get(m,c,"grouped"))])
    table(rows,[230,135,135]);p("All values use all three seeds. ± denotes sample standard deviation in percentage points, not a confidence interval. Every final checkpoint is evaluated; no best-seed selection is used.","small")
    figure("ablation_comparison.png",500,240)
    p("Figure 1. Final test accuracy across the fixed matrix. Dots show individual seeds; error bars show sample SD.","small")
    h("4.1  What the comparison supports")
    p("The MLP performs well under both protocols. Its paired grouped-minus-ordered mean difference is +0.51 percentage points with SD 3.45; individual signs differ. The base Transformer has a larger positive mean difference (+19.19 points) but SD 27.13 and mixed signs. These three-seed comparisons do not support a universal ranking of partition difficulty.")
    p("The grouped Transformer falls to 0.16% mean test accuracy at 30% training data and 0.46% with no weight decay, while the respective ordered means are 80.85% and 38.79%. These consistently negative paired gaps show strong sensitivity in this specific fixed-budget setup. They do not isolate a universal causal effect of weight decay or exchange symmetry. Low grouped accuracy can coexist with partial fitting of training data; it is not evidence of complete memorization followed by failed grokking.")

    # Page 4.
    page();h("4.2  Learning dynamics and retrieval control")
    figure("learning_curves.png",500,325)
    p("Figure 2. Mean train and validation accuracy for base runs; shading is sample SD. Curves are read directly from logged measurements. MLP generalization improves alongside fitting; base Transformer training remains incomplete at the fixed endpoint.","small")
    figure("swap_controls.png",500,195)
    p("Figure 3. Swap overlap, a non-neural swap lookup and the base Transformer. Overlap excludes diagonal pairs; accuracies include every test pair, so these denominators are intentionally different.","small")
    p(f"The ordered base test set has {get('mlp','main','ordered')['overlap_mean']*100:.2f}% non-diagonal reversal overlap on average. The lookup memorizes training labels, returns the reversed pair's stored label when available, and otherwise guesses the most common training class. Its {get('mlp','main','ordered')['lookup_mean']*100:.2f}% ordered accuracy falls to {get('mlp','main','grouped')['lookup_mean']*100:.2f}% under grouping. This is a constructive example of protocol sensitivity; it does not establish that either neural network uses this lookup mechanism. Uniform-random chance is 1/97 ≈ 1.03%, but deterministic learned predictions need not attain that benchmark.")

    # Page 5.
    page();h("5  Exploratory horizon extension")
    p("Because training accuracy showed incomplete base Transformer fitting, all six base Transformer settings were re-run to 12,000 updates with unchanged hyperparameters. This amendment occurred after some original test results had been seen. It is explicitly exploratory, reuses the same partitions and provides no independent confirmatory test. Re-running from the same seeds reproduces the first 3,000 logged updates exactly in all six runs.")
    table([["Horizon / base Transformer","Ordered test (%)","Grouped test (%)"],
           ["3,000 (fixed matrix)",fmt(get("transformer","main","ordered")),fmt(get("transformer","main","grouped"))],
           ["12,000 (exploratory)",fmt(get("transformer","horizon","ordered")),fmt(get("transformer","horizon","grouped"))]],[230,135,135])
    figure("extended_dynamics.png",500,215)
    p("Figure 4. Longer-horizon train/validation means and sample SD; the dotted line marks the original endpoint. The endpoints improve unevenly across partitions. Some trajectories fluctuate substantially.","small")
    h("5.1  Grokking outcome")
    p(f"Across {stats['runs']} runs, {stats['train_99_count']} reached the logged 99% training threshold and {stats['val_90_count']} reached 90% validation accuracy. None satisfied the predeclared ≥500-update delay criterion. This means the selected diagnostic was not observed here; it does not contradict published grokking results, prove absence at longer horizons, or validate an alternative mechanism. Finite-budget model comparisons and a negative dynamics result are the substantive outcomes of this study.")
    h("5.2  Error analysis")
    p("Every run retains each test input, target, prediction and whether its reversed pair appeared in training. Separate accuracies are computed for reversal-seen and reversal-unseen test examples; empty strata have an explicit null accuracy. Prediction swap consistency is also stored. Test-only error maps use a predetermined seed 0 for both base Transformer partitions and appear in figures/test_error_maps.png. Gray pixels mean the pair was not evaluated in test. The maps are descriptive and are not a causal probe of network representations.")

    # Page 6.
    page();h("6  Discussion")
    p("Evaluation defines what generalization means. New ordered pairs and new swap orbits ask different questions, and the retrieval control demonstrates why their scores can differ. Yet grouping does not always lower neural accuracy: model optimization and data coverage interact with partition composition. The near-ceiling MLP results support successful held-out prediction in both schemes, without identifying its internal algorithm. The Transformer ablations reveal a regime worth probing with better-matched controls and longer, stable optimization.")
    h("7  Limitations and next experiments")
    p("Only one modulus, two small architectures and three seeds per setting were tested. Seeds jointly vary initialization and partition membership; there is no crossed seed design separating those sources of variance. Parameter counts differ, and matching update counts does not match architecture-specific compute. Orbit grouping changes the number of distinct mathematical equivalence classes represented in training, plus class and diagonal composition. No Fourier interventions, causal ablations, label-balanced splits or commutative-by-design model were run. The extension is post-hoc and cannot be presented as fresh confirmatory evidence.")
    p("A useful follow-up would first freeze new seeds and horizons, add crossed model/partition seeds, separate diagonal pairs, and control label and orbit coverage. Train a permutation-invariant baseline alongside an ordered model to test architectural symmetry. Fourier spectra and interventions could then be used to investigate the mechanisms behind reproducible delayed generalization.")
    h("8  Reproducibility")
    env=json.loads((root/"results/mlp_main_ordered_s0/environment.json").read_text())
    p(f"The package contains all {stats['runs']} configurations, raw histories, split arrays/hashes, environment records, final weights, test predictions, per-run and aggregate tables, tests, plot/report builders and a verification command. The artifact verifier reloaded every checkpoint and matched its predictions to saved test arrays. {stats['summed_run_seconds']/60:.2f} minutes is the sum of recorded per-run wall times; workers ran concurrently, so it is not total elapsed time. The CPU environment used Python {env['python']}, PyTorch {env['torch']} and NumPy {env['numpy']} (authoritative per-run versions are in environment.json). The Colab helper is supplied but was not executed in Colab.")
    h("References")
    p("[1] A. Power, Y. Burda, H. Edwards, I. Babuschkin and V. Misra. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177, 2022. https://arxiv.org/abs/2201.02177","small")
    p("[2] N. Nanda, L. Chan, T. Lieberum, J. Smith and J. Steinhardt. Progress measures for grokking via mechanistic interpretability. ICLR 2023; arXiv:2301.05217. https://arxiv.org/abs/2301.05217","small")
    p("[3] A. Gromov. Grokking modular arithmetic. arXiv:2301.02679, 2023. https://arxiv.org/abs/2301.02679","small")
    build(docs/"Research_Report.pdf",story)
    (docs/"Research_Report.md").write_text("# Beyond Memorization\n\n"+"\n".join(md))

    print("Generated docs/Research_Report.pdf and docs/Research_Report.md")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args=ap.parse_args()
    create(args.root)


if __name__=="__main__":
    main()
