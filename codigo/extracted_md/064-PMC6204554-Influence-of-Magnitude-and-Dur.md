# Página 1

RESEARCH ARTICLE
Zebrafish Bone and General Physiology Are
Differently Affected by Hormones or
Changes in Gravity
Jessica Aceto1, Rasoul Nourizadeh-Lillabadi2, Raphael Marée3, Nadia Dardenne4,
Nathalie Jeanray1, Louis Wehenkel3, Peter Aleström2, Jack J. W. A. van Loon5,6☯,
Marc Muller1☯*
1 Laboratory for Organogenesis and Regeneration, GIGA- Research, University of Liège, B-4000, Liège,
Sart-Tilman, Belgium, 2 BasAM, Norwegian University of Life Sciences, Vetbio, 0033 Dep, Oslo, Norway,
3 GIGA & Department of Electrical Engineering and Computer Science, University of Liège, Liège, Belgium,
4 Unité de soutien méth. en Biostatistique et Epidémiologie, University of Liège, B23, Sart Tilman, Liège,
Belgium, 5 DESC (Dutch Experiment Support Center), Department of Oral and Maxillofacial Surgery / Oral
Pathology, VU University Medical Center & Academic Centre for Dentistry Amsterdam (ACTA), Amsterdam,
The Netherlands, 6 ESA-ESTEC, TEC-MMG, NL-2200 AG, Noordwijk, The Netherlands
☯These authors contributed equally to this work.
* m.muller@ulg.ac.be
Abstract
Teleost fish such as zebrafish (Danio rerio) are increasingly used for physiological, genetic
and developmental studies. Our understanding of the physiological consequences of al-
tered gravity in an entire organism is still incomplete. We used altered gravity and drug treat-
ment experiments to evaluate their effects specifically on bone formation and more
generally on whole genome gene expression. By combining morphometric tools with an ob-
jective scoring system for the state of development for each element in the head skeleton
and specific gene expression analysis, we confirmed and characterized in detail the de-
crease or increase of bone formation caused by a 5 day treatment (from 5dpf to 10 dpf) of,
respectively parathyroid hormone (PTH) or vitamin D3 (VitD3). Microarray transcriptome
analysis after 24 hours treatment reveals a general effect on physiology upon VitD3 treat-
ment, while PTH causes more specifically developmental effects. Hypergravity (3g from
5dpf to 9 dpf) exposure results in a significantly larger head and a significant increase in
bone formation for a subset of the cranial bones. Gene expression analysis after 24 hrs at
3g revealed differential expression of genes involved in the development and function of the
skeletal, muscular, nervous, endocrine and cardiovascular systems. Finally, we propose a
novel type of experimental approach, the "Reduced Gravity Paradigm", by keeping the de-
veloping larvae at 3g hypergravity for the first 5 days before returning them to 1g for one ad-
ditional day. 5 days exposure to 3g during these early stages also caused increased bone
formation, while gene expression analysis revealed a central network of regulatory genes
(hes5, sox10, lgals3bp, egr1, edn1, fos, fosb, klf2, gadd45ba and socs3a) whose expres-
sion was consistently affected by the transition from hyper- to normal gravity.
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
1 / 42
OPEN ACCESS
Citation: Aceto J, Nourizadeh-Lillabadi R, Marée R,
Dardenne N, Jeanray N, Wehenkel L, et al. (2015)
Zebrafish Bone and General Physiology Are
Differently Affected by Hormones or Changes in
Gravity. PLoS ONE 10(6): e0126928. doi:10.1371/
journal.pone.0126928
Academic Editor: Paul Eckhard Witten, Ghent
University, BELGIUM
Received: December 18, 2014
Accepted: April 9, 2015
Published: June 10, 2015
Copyright: © 2015 Aceto et al. This is an open
access article distributed under the terms of the
Creative Commons Attribution License, which permits
unrestricted use, distribution, and reproduction in any
medium, provided the original author and source are
credited.
Data Availability Statement: Raw data and
complete lists of analyzed data are publicly available
at Arrayexpress (https://www.ebi.ac.uk/arrayexpress/)
under the accessions: E-MTAB-3285, E-MTAB-3286,
E-MTAB-3289, and E-MTAB-3290.
Funding: This work was supported by the "Fonds de
la Recherche Fondamentale Collective"; 2.4555.99/
2.4542.00/2.4561.10, the SSTC; PAI: P5/35, the
University of Liège; GAME project, the European
Space Agency projects AO-99-LSS-003 and AO-99-
LSS-006, the Belgian Space Agency Prodex projects
FISH-GSIM and FISH-SIM. JvL received grant MG-


# Página 2

Introduction
For many years, the zebrafish has been recognized as an excellent model system for vertebrate
developmental biology. More recently, it is increasingly used to study vertebrate physiology,
pathology, pharmacology and toxicology [1–5]. Its main advantages are easy maintenance,
high fertility, rapid and external development, easy observation of all developmental stages,
small size, transparency of the embryos and close contact with surrounding medium (water) al-
lowing easy administration of drugs. In addition, its genome is sequenced and extensively an-
notated together with well established forward and reverse functional genomics and access to
already generated and characterized mutants and transgenic lines of fish (zfin.org).
Skeletal development in zebrafish was first more widely addressed in large scale mutagenesis
screening initiatives, resulting in identification of a number of genes required for early forma-
tion of the head skeleton [6, 7]. Cranial cartilage is the first skeletal structure to be detected as
early as 3 days post-fertilization (dpf), while first calcified intramembranous bone structures
start to form at about the same time. Perichondral bone elements slowly build up on the exist-
ing cartilage matrix during the following days. In mammals, one of the major genes involved in
osteoblast differentiation is Runx2. In zebrafish, its ortholog runx2b is similarly required for os-
teoblast differentiation [8] and the onset of osteoblast specific genes [9], such as members of
the dlx family [10] and osterix (osx) [11, 12], again with mammalian orthologs. Other expressed
genes code for bone extracellular matrix (ECM) proteins osteocalcin (Osc2)[13], collagen10a1a
(Col10a1a)[14], Bglap, Spp1 and collagen1a1a (Col1a1a) [9, 15, 16]. The latter is mutated in
the chihuahua (chi) mutant, a model for the human condition of osteogenesis imperfecta. Final-
ly, correct calcification of the bone ECM depends on transcellular epithelial calcium uptake
through the calcium channel Trpv5/6 [17] and the precise control of phosphate/pyrophosphate
homeostasis by the Entpd5 diphosphohydrolase, expressed in osteoblasts [18] together with
the widely expressed phosphodiesterase Enpp1 [19]. Taken together, these observations indi-
cate an extensive similarity of the molecular pathways governing bone physiology between tele-
osts and mammals, validating the zebrafish as a vertebrate model in this field [16, 20–22].
During space flight, human passengers experience profound alterations of their skeletal and
muscular system, as well as blood circulatory and immune systems [23–25]. Microgravity is
the main differential factor of the environment in space and is probably responsible for the
rapid bone loss (osteoporosis) observed in space. Various fish species, such as carp [26], gold-
fish [27–31], or cichlids [32–39] have been utilized in the past for evaluating the effects of al-
tered gravity. More recently, smaller fishes such as swordtail [37, 40], medaka [41–46] and
zebrafish [47–51] have attracted more attention. Most analyses using fishes have concentrated
on the impact of altered gravity on graviperception [33, 52], the vestibular system [37, 53, 54]
and its involvement in motion sickness [38, 55–57]. Several studies also revealed that general
embryogenesis of various organisms is not affected by gravity conditions (review in [46, 49, 50,
58]).
Here, we investigate the effect of increased gravity on the general physiology of zebrafish lar-
vae by using a Large Diameter Centrifuge (LDC) [59] to study whole genome gene expression.
We investigate in more detail the effects on head skeleton development and we validate our ap-
proach by studying the effects of drug treatments (VitD3 and PTH) known to affect bone for-
mation. Finally, we propose a novel approach to study the effects of microgravity by growing
zebrafish in hypergravity for 5 days (from 0–5dpf) before returning them to normal gravity,
the Reduced Gravity Paradigm, RGP [60]. The hypothesis for this paradigm dictates that simi-
lar effects as observed from the transition going from 1g into micro-g are observed going from
a hypergravity level towards a 1g acceleration, a special kind of simulated microgravity or
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
2 / 42
057 from the Netherlands Organisation for Scientific
(NWO) Research Earth and Life Sciences via the
Netherlands Space Office NSO.
Competing Interests: Co-author Marc Muller is a
PLOS ONE Editorial Board member. This does not
alter the authors' adherence to PLOS ONE Editorial
policies and criteria.


# Página 3

‘relative microgravity’. However, it may be expected that the magnitude of the effects applying
RGP is reduced as compared to the 1g - μg transition.
Materials and Methods
Animal procedures
Zebrafish (Danio rerio) were maintained under standard conditions [61] in the GIGA zebrafish
facility (licence LA2610359). Briefly, zebrafish (Danio rerio) of the AB strain were reared in a
recirculating system from Techniplast, Italy at a maximal density of 7 fish/l. The water charac-
teristics were as follows: pH = 7.4, conductivity = 500 μScm-1, temperature = 28°C. The light
cycle was controlled (14 h light, 10 h dark). Fish were fed twice daily with dry powder (ZM fish
food) adapted to their age and once daily with fresh Artemia salina nauplii (ZM fish food). Lar-
vae aged less than 14 days were also fed twice daily with a live paramecia culture. Wild type em-
bryos were used and staged according to [62].
The day before breeding, wild-type adult male and female zebrafish were set up in several
breeding tanks, separated by a clear plastic wall. After the light was turned on the next morn-
ing, walls are removed, eggs are generated by natural mating and collected from 30 minutes to
2 hours after spawning. After sorting, clean eggs are moved to Petri dishes and incubated at
28°C in E3 medium (5 mM Na Cl, 0.17 mM KCl, 0.33 mM CaCl2, 0.33 mM MgSO4, 0.00001%
Methylene Blue). All protocols for experiments were evaluated by the Institutional Animal
Care and Use Committee of the University of Liège and approved under the file numbers 568,
1074, and 1264 (licence LA 1610002).
Chemicals
Parathyroid hormone (PTH; Merck-Calbiochem, Overijse, Belgium) stock solution (1μg/ml)
was prepared in DMSO and stored in aliquots at -20°C. Vitamin D3 (cholecalciferol, VitD3;
Sigma-Aldrich, Diegem, Belgium) stock solution (200μl/ml) in DMSO was stored in aliquots at
-20°C for maximum one month.
Chemical treatments
The chemical protocol was inspired by Fleming and collaborators experiments [63]. Larvae at
5dpf were transferred into a 6 well plate (Millipore) containing E3 medium supplemented with
the required chemical or vehicle (DMSO) as negative control. The medium was changed every
day at the same time. Final concentrations in E3 were at 10ng/ml for PTH and 200ng/ml for
VitD3. Each well contained 20 fish in 4ml. They were treated for 1day (n = 50–60 larvae) to
perform microarrays and for 5days, from 5 to 9 or 10dpf, to observe the longer-term effects of
treatments by different staining (n = 20–30 larvae). Plates were placed into the dark and incu-
bated at 28°C. The larvae were euthanized by tricaine overdose (0.048% w/v) and directly sub-
mitted to an RNA extraction at 6dpf (for microarrays) or a 4% para-formaldehyde (PFA;
Sigma-Aldrich, Diegem, Belgium) fixation at 6, 9 or 10dpf (for staining).
Hypergravity experiments in the Large Diameter Centrifuge
A Large Diameter Centrifuge (LDC) was used for hypergravity experiments. It is composed of
a central axis linked to 2 perpendicular arms, each arm terminating in 2 opposing gondolas
where it is possible to install an incubator containing the samples. The arms provide an 8m di-
ameter for rotation and can provide centrifugal forces of maximum 20g. The zebrafish larvae
were incubated in 20 ml E3 in a Petri dish placed in an incubator within a gondola for 3g
experiments, and placed either in an incubator on the centrifuge axis (axe) or outside of the
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
3 / 42


# Página 4

centrifuge for 1g controls. In this setting, the medium represents less then 5 mm of water col-
umn and thus the 3g acceleration causes an increase in hydrostatic pressure of maximum
0.0015 bar, as compared to the 1bar atmospheric pressure [64].
Staining methods
Acid-free protocols were adapted [65] to perform Alcian blue (8 GX Sigma-Aldrich, Diegem,
Belgium) staining of cartilage structures and Alizarin red S (Sigma-Aldrich, Diegem, Belgium)
staining of calcified structures. At 6, 9 or 10dpf, the larvae were fixed in 4% PFA for 2h at room
temperature and rinsed several times with PBST.
Cartilage was stained overnight in 10 mM MgCl2, 80% EtOH and 0.04% Alcian blue. The
larvae were washed in different concentrations of ethanol (80%, 50%, 25%) to remove excess
staining. Pigmentation was bleached in a H2O2 solution (H2O2 3%, KOH 0.5%) and finally the
larvae were rinsed 3 times in a solution of 25% glycerol / 0.1% KOH and 50% glycerol, 0.1%
KOH and finally stored in this solution at 4°C.
During acid-free bone structure staining with Alizarin red, bleaching was performed imme-
diately after fixation, before the staining. After the bleaching, long rinses (at least 20min each)
in a 25% glycerol, 0.1% KOH solution are necessary to prevent the fading of the staining. The
larvae are stained in a 0.05% Alizarin red solution in water for 30min in the dark on low agita-
tion, rinsed in a 50% glycerol, 0.1% KOH solution to remove excess staining and kept at 4°C in
the same solution.
Images of stained larvae (n = 20–30 larvae) were obtained on a binocular (Olympus, cell B
software).
Image analysis
Image analysis was performed on the pictures of larvae stained with Alcian blue for cartilage or
Alizarin red for bone. Individual cartilage and bone elements were identified according to [10,
15, 66–68]. For morphometric analysis, images were uploaded into the CYTOMINE environ-
ment [69] and manually annotated by positioning 21 landmarks for larvae stained for cartilage
(Fig 1A) as previously defined in the CYTOMINE ontology. 29 landmarks were placed for lar-
vae stained for bone in hormonal treatments (Fig 1C), of which 15 were selected for the hyper-
gravity experiments. The program then defines the positions of all selected landmarks and
computes all the distances (in pixels) and angles (in radian) of all the possibilities between two
points of interest. These data were exported into an Excel file and a selection of interesting
measures was conducted by performing principal component analysis on data obtained from
differently treated larvae to identify invariable or redundant measures. The measures selected
were: for cartilage (Alcian blue): Anterior to Ethmoid plate, Anterior to Posterior, Articulation
down to Articulation up, Ceratohyal ext. down to Ceratohyal ext. up, Ceratohyal ext. down to
Ceratohyal int. down, Ceratohyal ext. up to Ceratohyal int. up, Ethmoid plate to Posterior,
Hyosymplectic down to Hyosymplectic up; and for bone (Alizarin red): Anguloarticular down
to Anguloarticular up, Anterior to Notochord, Anterior to Parasphenoid a, Branchiostegal ray
1 down to Branchiostegal ray 1 up, Entopterygoid down to Entopterygoid up, Maxilla down to
Maxilla up, Opercle down to Opercle up, Parasphenoid a to Parasphenoid b, Parasphenoid b to
Parasphenoid c, area of the parasphenoid triangle: parasphenoid a, b, and c, and finally the an-
gles between parasphenoid a and b, a and c, b and c.
Statistics were performed using GraphPad Prism5. A t-test was used for control versus treat-
ment experiments, while a one way ANOVA was used for multiple comparisons.
Morphometric analysis did not inform about the extent of ossification within each larva.
Thus, a systematic structure analysis was generated. Each bone structure was classified based
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
4 / 42


# Página 5

Fig 1. (A-D) Cartilage and bone elements of the head skeleton in 10dpf zebrafish. (A) Alcian blue staining of head cartilage representing the landmarks
used for morphometry. (B) Schematic representation of the different head cartilage elements. anterior limit (an), articulation (ar), ceratobranchial pairs 1 to 4
(cb1-4), ceratohyal (ch), ethmoid plate (et), hyosymplectic (h), Meckel's cartilage (mk), palatoquadrate (pq), posterior limit (po). (C) Alizarin red staining of
cranial bones representing the landmarks used for morphometry. (D) Schematic representation of the different cranial bone elements with 29 landmarks used
for chemicals treatments and 15 landmarks for the 3g and the relative-hypergravity. The 15 landmarks are anguloarticular (aa), anterior (an), branchiostegal
ray1 (br1), entopterygoid (en), maxilla (m), notochord (n), opercle (o), parasphenoid (p). Note that the parasphenoid is a triangular bone defined by its anterior
summit (a) and two posterior summits (b,c). The 29 landmarks include the 15 named before with branchiostegal ray2 (br2), cleithrum (c), ceratobranchial 5
(cb), ceratohyal (ch), dentary (d), hyomandibular (hm). (E-J) 10dpf zebrafish larvae after 5 days chemical treatments. (E-G) Alcian blue staining of
cartilage. (H-J) Alizarin red staining of bone. (E,H) Controls in DMSO. (F,G) no significant effect of, respectively VitD3 and PTH on cartilage development, nor
on chondrocyte shape or size (inlays showing close-up). I: increase of bone development after VitD3 treatment. (J) decrease of bone development after PTH
treatment. Ventral views, anterior to the left, (E-J) scale bar = 250μm.
doi:10.1371/journal.pone.0126928.g001
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
5 / 42


# Página 6

on the progress of development into one of the four following categories: absent, early ossifica-
tion, advanced ossification and over ossification. When values were considered as quantitative,
comparison between two groups (control versus chemical treatment or hypergravity in 1g>3g)
was assessed by a Student t-test, while comparison between different treatments ("relative mi-
crogravity" experiment) was assessed by an analysis of variance (ANOVA). A contingency
table considered ordinal values distributed among the 4 classes (from absent to over ossifica-
tion) or only 3 classes when one class was not present in the sample. Association between clas-
ses and treatment was assessed by X² test and by an ordinal logistic regression and the odds
ratio (OR). The "relative microgravity" experiment was analyzed in addition by grouping the
3g, 3g>1g and 3g>axe versus the 1g sample.
Statistical analyses were performed using the Statistica Software (version 10). Results were
considered statistically significant at the 5% critical level (p < 0.05).
RNA extraction and reverse transcription
Larvae at 6dpf, after 24h treatment, were used for the RNA extraction. Total RNA was ex-
tracted of 60 larvae per experiment using Trizol, followed by the RNeasy Mini kit (Qiagen, Hil-
den, Germany) according to the manufacturer’s instructions and conserved at -80 degrees
using. They were treated with Rnase-free Dnase Set (Qiagen, Hilden, Germany). After extrac-
tion, the quality and concentration of total RNA was evaluated by electrophoresis on capillary
gel and the ratio of absorbance at 260/280nm by spectrophotometer (Bioanalyzer 2100, Agilent
Technologies, Diegem, Belgium). Synthesis of cDNA was performed from 1μg of total RNA,
which was reverse transcribed (Transcriptor iScript cDNA Synthesis Kit, Bio-Rad, Nazareth,
Belgium) according to the manufacturer’s instructions.
Real Time-PCR
Gene-specific oligonucleotide primers were designed using Primer3 software to span exon-
exon junctions to avoid detection of genomic DNA contamination (see S1 Table for primer se-
quences) and synthesized by Eurogentec (Seraing, Belgium) or Integrated DNA Technology
(Leuven, Belgium). cDNA was used as template for quantitative Real-Time PCR with the Sensi-
Mix SYBR Kit (Bioline, London, UK), containing Sybr green. Reactions were performed on an
Applied Biosystems 7900 HT sequences Detection System (Applied Biosystems, Foster City,
CA) using the onboard software (SDS 2.4). Purity of the amplicons was checked by melting
curves at the end of each reaction. Ct values were exported from the onboard software as a text
file and imported into a customized Microsoft excel spreadsheet. 1 μl of the RT reaction (1/20
of the total cDNA) was added to 1X SYBR green master mix (Bioline, London, UK), 150 nmol
of each primer in 15 μl total volume. Samples were run in triplicate in optically clear 384-well
plates (ABgene), sealed with optical adhesive film (Applied Biosystems). "No template" con-
trols were run for all reactions, and all RNA preparations were subjected to sham reverse tran-
scription to check for the absence of genomic DNA amplification. The relative transcript level
of each gene was obtained by the 2-ΔΔCt method [70] and normalized relative to the gapdh
(glyceraldehyde-3-phosphate deshydrogenase) housekeeping gene chosen from a panel of 3
genes (gapdh, ef1-a, ß-actin) as the most stably expressed throughout our experiments (not
shown). Data from biological replicates were averaged and shown as mean normalized gene
expression ± SD.
Cycling parameters: 50°C x 2 min, 95°C x 10 min, then 40 cycles of the following 95°C x
15 s, 62°C x 20 s. A melting temperature-determining dissociation step was performed at
95°C x 15 s, 60°C x 15 s, and 95°C x 15 s at the end of the amplification phase.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
6 / 42


# Página 7

Microarray expression experiments
For microarray expression analysis, four replicates from each treatment (control and drug or
gravity treatment) were analyzed in 2+2 dye-swap hybridizations. One μg total RNA was line-
arly amplified one round and labeled, using Amino Allyl Message Amp II aRNA amplification
kit (Ambion-Life Technologies, Gent, Belgium) as previously described [71]. Five μg of the re-
sulting antisense RNA (aRNA) from the exposed and control groups was labeled either with
Cy3-dUTP or Cy5-dUTP (GE Healthcare Bio-Sciences AB, Uppsala, Sweden). The labeled tar-
gets were examined for amplification yield and incorporation efficiency by measuring the
aRNA concentration at 260 nm, Cy3 incorporation at 550 nm, and Cy5 at 650 nm using Nano-
drop (Thermoscientific, Wilmington, DE, USA). A good aRNA probe had a labeling efficiency
of 30–50 fluorochromes every 1000 bases. One to 5 μg of each labeled aRNA target was mixed,
9 μl 25× fragmentation buffer (Agilent Technologies, Diegem, Belgium) added, and the final
volume adjusted to 225 μl with RNase-free H2O followed by incubation for 30 min at 60°C.
The hybridization solution was prepared by adding 220.5 μl of 2× hybridization buffer (Agilent
Technologies, Diegem, Belgium) and 4.5 μl sonicated herring sperm DNA (10 μg/μl; Promega,
Madison, WI, USA) to the labeled target aRNA. Microarray slides (4x44K zebrafish V2 or V3,
Agilent Technologies, Diegem, Belgium) were prehybridized at 42°C, 60 min using 0.1% bovine
serum albumin (BSA) Fraction V, 5× SSC, and 0.1% sodium dodecyl sulfate (SDS). Hybridiza-
tion was performed at 60°C in 16 h using gasket slides, hybridization chamber, and oven (Agi-
lent Technologies, Diegem, Belgium) according to Agilent 60-mer oligo microarray processing
protocol. Microarray slides were then washed 3 × 5 min in 0.5 × SSC, 0.01% SDS (first wash at
42°C and next two at room temperature). Finally, slides were washed 3 times in room temp
with 0.06× SSC and dried immediately with centrifugation at 800×g for 1 min.
Microarray slides were scanned using a GenePix 4000B (Axon instrument, Foster City, CA).
Scanning was performed at a level just before saturation of several spots. Raw data generated
from Genepix were imported into the Bioconductor package LIMMA and corrected for back-
ground [72]. For within-array and between-array normalization, print tip Loess and scale were
used, respectively [72]. An empirical Bayes moderated t-test [72, 73] was applied to detect dif-
ferently expressed genes across treated and control samples. The p values were corrected for
multiple testing using the Benjamini–Hochberg (BH) [74] method and p-values <0.1 were se-
lected as differently expressed genes. The generated gene list was further filtered for genes with
low intensity and with small changes in expression. In the averaged normalized MA-Plot, the
majority of genes were clustered in between M values of ±0.4 (fold change ±1.3) and selected
to be threshold criteria for differently expressed gene list. The VitD3 data were obtained on a
SureScan Dx instrument (Agilent Technologies, Diegem, Belgium) and analyzed using the
GeneSpring software (Agilent Technologies, Diegem, Belgium) by applying the same settings.
Raw data and complete lists of analyzed data are publicly available at Arrayexpress (https://
www.ebi.ac.uk/arrayexpress/) under the accessions: E-MTAB-3285, E-MTAB-3286, E-MTAB-
3289, and E-MTAB-3290.
Ingenuity Pathway Analysis
For pathway and biological function analysis of significantly differently expressed genes,
Ingenuity pathway analyses (IPA, QIAGEN Redwood City; http://www.ingenuity.com) were
used. The lists with differently expressed genes generated by the microarray analysis were
translated into mammalian (human, mouse, and rat) orthologs using the Unigene & Gene
Ontology Annotation Tool and uploaded to IPA. The IPA software is an online exploratory
tool with a curated database for over 20,000 mammalian genes and 1.9 million published litera-
ture references. IPA’s database together with EntrezGene, Gene Ontology, etc., integrates
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
7 / 42


# Página 8

transcriptomics data with mining techniques to predict and build gene networks, pathways,
and biological function clusters. The output results are given scores and p-values that are com-
puted based on the number of uploaded genes in the cluster or network and the size of the net-
work or cluster in the Ingenuity knowledge database. Fisher’s exact test is used to determine
the probability that each associated biological function is due to chance alone. Scores for IPA
networks are the negative logarithm of the p-value, indicating the likelihood of the focus genes
(genes uploaded to IPA) in a network being found together due to random chance. Scores of 2
or higher have at least a 99% likelihood of not being generated by chance alone.
Results
Effects of drug treatments on head skeletal formation
To characterize in detail the process of cartilage and bone formation in zebrafish, we first
wanted to examine the effects of chemical treatments known to affect skeletal development.
Treatment of zebrafish larvae with vitamin D (VitD3) was previously shown to result in en-
hanced bone formation, while continuous treatment with parathyroid hormone (PTH) led to
decreased bone formation [63]. We decided to confirm and extend these findings by compar-
ing the effects on skeletal formation to those on gene expression.
VitD3 and PTH treatments were performed continuously from 5dpf to 10dpf. Control and
treated larvae were stained by Alcian blue for cartilage extracellular matrix (ECM) and with
Alizarin red to detect the calcified bone matrix. At this stage, the head cartilage is well formed
and a complete set of cartilage elements is observed (Fig 1A and 1B). In contrast, although ossi-
fication begins at 3dpf and the first bone structures are visible at 5dpf, the bone skeleton con-
tinues its formation until 30dpf [68]. Nevertheless, at 10dpf, a number of bone elements are
observed in the head region, the first vertebral centrae are formed, while others only begin to
be calcified (for example the branchiostegal ray2) (Fig 1C and 1D).
In three independent experiments, 27–29 ventral view images of Alcian blue- or Alizarin
red-stained larvae were obtained. After 5days of VitD3 or PTH treatment, cartilage stays un-
changed as compared to the control by general observation. The structures are well formed,
complete with the glycosaminoglycans present in the cartilage matrix judging from the similar
staining intensity (Fig 1E–1G). In a close-up view (Fig 1E–1G, inlays), no difference could be
observed in cell shape or size between the different treatments. Considering bone calcification,
a general observation revealed a clear increase of bone development upon VitD3 treatment
(Fig 1I). Some structures appear in advance, such as the retroarticular (Fig 1I arrowhead) bone
and the preopercular (not shown) bone, while some other structures are thicker such as the
dentary or the ceratohyal, or longer such as the branchiostegal ray2. Nevertheless, the general
morphology was unchanged. In contrast, continuous PTH treatment led to a general decrease
of bone formation and to a complete absence of some structures, such as the anguloarticulars
and branchiostegal ray2 (Fig 1J).
Based on these images, we applied two complementary approaches to obtain a more objec-
tive qualitative and quantitative description of the skeleton. The first one is a morphometric
approach that evaluates the general aspect of the head skeleton by measuring the distances be-
tween and the relative position of all detected bone elements. The images were introduced into
the CYTOMINE software (see Materials and Methods, [69]) and each image was annotated by
positioning specific landmarks representing the different skeletal elements. For larvae stained
for cartilage, 21 landmarks were defined (Fig 1A), while 29 points of interest were positioned
within the Alizarin red-stained bone skeleton (Fig 1C). In these pictures, we consider the head
separated horizontally in 2 parts. Some structures are unique and located on the symmetry
axis, while others are paired and localized symmetrically, such as the dentary, maxilla,
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
8 / 42


# Página 9

entopterygoid, and hyosymplectic. To facilitate recognition, these were labelled “up” and
“down”. The software then computes the distances between selected landmarks and the angles
formed by lines drawn between selected points.
Morphometric analysis in VitD3-treated larvae cartilage revealed an increase of the distance
between articulation (ar) "up" and "down", leading to a broader jaw as compared to untreated
animals, while all the other distances remained unchanged (S1A and S1C Fig). Morphometric
cartilage analysis of larvae treated with PTH for 5 days revealed an increase in length of the cer-
atohyal cartilages (ch, S1B Fig). Analysis of the bone skeleton after VitD3 treatment revealed a
significant increase of the distance between maxillae (m, Fig 2A), consistent with a broader jaw
as already observed by cartilage morphometry. The length of the head skeleton is also increased
upon VitD3 treatment with a longer distance between the anterior part of the head (an) and
the notochord (n) or the parasphenoid (p). Other measures are not significantly modified (Fig
2A and 2C). PTH treatment led to a significant decrease of the size of the parasphenoid (p, Fig
2C). Some structures are missing, such as the anguloarticular (aa), branchiostegal ray2 (br2),
ceratohyal (ch) and/or maxilla (m) and a significant broadening of the posterior head skeleton
is revealed by the increased distance between left and right ("up" and "down") branchiostegal
rays1 (br1), entopterygoids (en), and opercula (o) (Fig 2B).
The second approach consists in the evaluation of the intensity and progression of bone for-
mation of the different bone structures, and their level of ossification. In each image, every
bone structure is assigned a score, ranging from absent (red), early ossification (yellow), ad-
vanced (green) or over-ossified (purple) in comparison to a typical image of a control larva of
the same age. The distribution of the scores obtained for the different elements in VitD3- or
PTH-treated larvae and the corresponding controls is shown in Fig 3 and the results of the sta-
tistical analysis are given in S2 and S3 Tables.
After 5 days VitD3 treatment, all the structures are present and some are over-ossified like
the hyomandibular, the entopterygoid, the dentary and the ceratohyal bones. Early (delayed)
ossification is decreased for all the structures shown, as compared to controls, while advanced
ossification increased in the maxilla, branchiostegal ray1, branchiostegal ray2 and anguloarti-
cular (Fig 3A). Statistical analysis (S2 Table) reveals that only the anguloarticular and the max-
illa up do not change significantly in this condition. All the other structures (br1, br2, m down,
ch, d, en, hm) are significantly increased, with the hyomandibulars, entopterygoids and cera-
tohyals displaying the most drastic effect. These results confirm a very significant positive effect
of VitD3 treatment on bone formation.
PTH treatment resulted in nearly opposite effects to VitD3. Only the entopterygoid and the
branchiostegal ray1 are present in each fish (Fig 3B) with the branchiostegal ray1 unaffected
and the entopterygoid displaying 60% of early ossification in PTH-treated larvae compared to
3,45% in controls. All the other structures were absent in at least 20% of the total 27 fish ana-
lyzed. The strongest effect was seen in the anguloarticular bone with 94% of absence compared
to 21% absence, 19% early ossification and 60% of advanced ossification in the controls. Specif-
ic statistical analysis confirmed that PTH treatment significantly (p<0,001) reduced nearly all
the structures except branchiostegal ray1 (S3 Table).
To obtain a global score describing the head skeleton in the different conditions, the individ-
ual structure scores in each image were added up and a mean global score was obtained show-
ing that VitD3 treatment significantly increases bone development (from a score of 26±3 in the
controls to 33±4 in the VitD3 treatment), while PTH treatment significantly decreases ossifica-
tion to approximately half of untreated control (from a score of 27± 4 to 13±5,5).
In summary, these complete image analyses reveals that VitD3 treatment conserves the gen-
eral skeletal morphology, but leads to a longer head and a larger jaw. Bone calcification is stron-
ger for most elements, and some elements calcify earlier. In contrast, PTH treatment conserves
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
9 / 42


# Página 10

Fig 2. Morphometric analysis results of bone matrix staining after 5 days chemical treatments. The distances are measured in pixels. Mean ± SD and
t-test analysis were calculated for each measure on at least 20 individuals. * p < 0.05, ** p < 0.01 and ***p < 0.001. (A) Distances after VitD3 treatment. (B)
Distances after PTH treatment. (C) Area of the parasphenoid bone results after 5 days PTH or VitD3 treatment. Abbreviations as in Fig 1. A) Analysis of the
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
10 / 42


# Página 11

the general cartilage morphology except for an increased length of the ceratohyal. In bone,
PTH treatment leads to a general decrease of ossification. Some structures are missing and the
parasphenoid is significantly decreased.
Modification of gene expression upon drug treatment
To gain deeper insight into the molecular mechanisms involved in the observed skeletal modi-
fications, we analyzed the expression of several genes selected for their known function in bone
formation. One class of genes codes for structural proteins such as collagens (Col1a1, Col1a2,
Col10a1a) or bone specific ECM proteins such as secreted acidic cysteine rich protein (Sparc,
previously named osteonectin or Osn), secreted phosphoprotein 1(Spp1, previously named
osteopontin or Osp) and bone gamma-carboxyglutamate protein (Bglap, previously named
osteocalcin or Ocn). The second class of interest consists of those genes coding for factors in-
volved in regulation of cartilage and bone differentiation, including the pth1a gene coding for
PTH as well as transcription factor genes dlx5a, dlx6a, runx2b and osx.
We first decided to follow the expression of these genes during the 6–10dpf period in un-
treated animals, using the glyceraldehyde-3-phosphate dehydrogenase (gapdh) house-keeping
gene as reference (selected from 3 candidate housekeeping genes, see Materials and Methods).
Compared to their expression at 6dpf, we observe an increase of sparc, bglap, spp1 and col1a1
at 7dpf, followed by a decrease at 8dpf for sparc, bglap and spp1, while the col1a1 gene peaked
at 8dpf and decreased its expression at later stages (Fig 4A and 4D). The pth1a gene expression
strongly increased 76-fold during the 6–10dpf period, while runx2b displayed a 10-fold in-
crease. The transcription factor gene dlx5a displayed an expression peak at 7 and 8dpf and de-
creased after that, while dlx6a was unaffected and osx surprisingly revealed a 2-fold decrease
from 6 to 7dpf (Fig 4G and 4J).
We then investigated the modulation of expression of these genes during drug treatment
starting at 5dpf. Compared to untreated controls, VitD3 treatment led to a clear and significant
increase in expression of all the structural protein genes: sparc, bglap, spp1, col1a1 and, to a less-
er extent col1a2 and col10a1a (Fig 4C and 4F). These results correlate well with the observed in-
crease in bone calcification observed at 10dpf. Among the regulatory factor genes, only pth1a
revealed a strong up-regulation that increased during the treatment, while dlx5a and dlx6a
were transiently induced at 8 and 9dpf. Finally, runx2b displayed a weak but significant in-
crease up to 1.5-fold at 10dpf, and osx1 was only transiently induced 2-fold at 7dpf (Fig 4I and
4J).
On the other hand, relative to untreated controls, PTH treatment resulted in a transient in-
crease of spp1 at 8–9dpf, while sparc, and bglap were unchanged before a decrease at 10dpf (Fig
4B). Surprisingly, no significant effect of PTH treatment was observed on the expression of the
collagen genes (Fig 4E). Among the regulatory factors, osx expression remained constant, while
pth1a, dlx5a, dlx6a and runx2b declined at 10 dpf (Fig 4H and 4K). Taken together, these ob-
servations are consistent with the observed decrease in bone matrix calcification at 10dpf.
bone skeleton after VitD3 treatment revealed a significant increase of the distance between maxillae (m), consistent with a broader jaw as already observed
by cartilage morphometry. The length of the head skeleton is also increased upon VitD3 treatment with a longer distance between the anterior part of the
head (an) and the notochord (n), and between an and the parasphenoid (p) bone. Other measures are not significantly modified (A, C). B) PTH treatment
caused an increase of the distance between the anterior part of the head and the summit “a” of the parasphenoid, mainly due to a significant decrease of the
size of the parasphenoid (p) (C). Some structures are missing, such as the anguloarticular (aa), branchiostegal ray2 (br2), ceratohyal (ch) and/or maxilla (m).
However, a significant broadening of the posterior head skeleton is revealed by the increased distance between left and right ("up" and "down")
branchiostegal rays1 (br1), entopterygoids (en) and also the opercula (o) (B).
doi:10.1371/journal.pone.0126928.g002
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
11 / 42


# Página 12

Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
12 / 42


# Página 13

Whole genome analysis of gene expression modulation by drugs
To obtain a global view of the physiological changes caused by PTH and VitD3 treatment, we
performed a microarray whole genome expression analysis. We compared 6dpf control larvae
to larvae treated between 5dpf and 6dpf with the corresponding compounds, in order to cap-
ture early regulatory events rather then secondary regulations leading ultimately to the ob-
served modulations of bone formation at 10dpf.
Four independent experiments were carried out and total RNA was extracted from control
and VitD3-treated 6dpf larvae. A complete list of genes affected more than 1.3-fold (log2 fold
change 0.4) by VitD3 treatment is given in S4 Table (p-value<0.1). Six genes were selected
from the list for validation by RT-qPCR, which demonstrated the reliability of the microarray
data (Table 1). Confirming that the VitD3 pathway was indeed activated, the most highly in-
duced gene is cyp24a1, encoding a member of the cytochrome P450 superfamily of enzymes in-
volved in the degradation of 1,25-dihydroxyvitamine D3. Modulation of the insulin pathway is
indicated by the significant induction of igfbp1 and igf2. According to Ingenuity Pathway Anal-
ysis (IPA; Materials and Methods), other biological functions that were affected by vitamin D
treatment (S5 Table) are related to lipid, small molecule, amino acid, carbohydrate and drug
metabolism, followed by organismal and cardiovascular system development. A striking fea-
ture of the affected genes list is the abundance of genes involved in molecular transport, from
ion channels to ATP-dependent pumps (S4 Table), consistent with a profound adaptation to
the changes in metabolism that were also previously observed [75–77]. Among the transcrip-
tion regulatory factors, we note the decreased expression of ppara and of foxo3, involved in
lipid metabolism, as well as fosb and twist1, while klf11 and klf13 were significantly induced (S4
Table). As these experiments were performed using mRNA from the entire larvae, we at-
tempted to focus on individual organ systems by filtering the affected gene set against available
databases of genes involved in muscle or cartilage/bone function (GO annotation of human
gene orthologs using IPA knowledge base). A network of regulatory interactions could be con-
structed, comprising genes common to both systems and genes specific for each organ (S2 Fig).
Major hubs, such as the protooncogene MYC controlling cell proliferation, components of the
insulin-like pathway such as IGFBP1 and IGF2, or the cytokine receptor regulator SOCS1 are
common to both systems. Specific to muscle, regulators such as PPARA or FOXO3 are down-
regulated, while STAT3, mediating the cytokine receptor response, is up-regulated. Interesting-
ly, muscle structural genes such as TTN (Titine) are inhibited. Other affected genes are bone-
specific transcription factors, such as ATF4 and FOSB, a member of the WNT pathway
(WNT3) or the carbohydrate (glycoprotein)-binding protein LECT1 (Lectin1).
PTH treatment between 5dpf and 6dpf resulted in less modulation of gene expression (S6
Table). Six genes were selected from the list to include up- and down-regulated genes for inde-
pendent confirmation of the microarray expression results by RT-qPCR (Table 1). Interesting-
ly, we observed a decrease (2.5-fold) in the expression of the endogenous pth1a gene (PTH in
S6 Table), thus confirming the previous RT-PCR results (Fig 4K) and suggesting that the PTH
Fig 3. Extent of bone formation in 10dpf larvae after 5days chemical treatments. Bone development is
classified for each element into different categories: Absent (no structure present; red), early ossification
(beginning of the bone ossification; yellow), advanced ossification (the structure is present and already
developed as the control; green) and over ossification (the structure is more developed compared to the
control; purple). Cumulated frequencies in % are represented for each element. As no significant difference
was observed for paired structures between left and right (up and down), their scores have been combined.
Statistical analysis was performed by X² of Pearson and a logistic regression. (A) Cumulated frequency after
5days VitD3 treatment. To obtain this, values were attributed to each element according to its category and
added up for each larva: 0 for absent, 1 for early, 2 for advanced, and 4 for over ossification (B) Cumulated
frequency after 5days PTH treatment.
doi:10.1371/journal.pone.0126928.g003
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
13 / 42


# Página 14

treatment was effective, as the larvae exerted a compensatory response by decreasing endoge-
nous PTH production. In rat and human osteoblastic cells, PTH receptor mRNA was shown to
be down-regulated upon PTH treatment [78, 79], in contrast we observe a significant induction
(1.9-fold) of PTH receptor (pth1rb), suggesting more complex regulatory networks in using an
in vivo model as opposed to in vitro cultures. Additional affected genes are the repressed
cyp21a2 and hsd3b7, indicating a decrease in steroid degradation. The increased expression of
rxra nuclear receptor mRNA (S6 Table) contrasts with the observed VitD3 effects (S4 Table),
Fig 4. Expression of bone-specific genes during development between 6 and 10dpf. (A,D,G,J) Specific mRNA levels at 6dpf relative to the gapdh
house-keeping gene were used as reference, and then compared to the corresponding level in larvae of different age. (E-F,H-I,K-L) Specific mRNA levels in
treated larvae were determined relative to the gapdh reference house-keeping gene and then compared to the corresponding level in untreated controls of
the same age. (A-C) Bone matrix markers bglap, sparc, spp1. (D-F) Collagens col1a1, col1a2, col10a1a. (G-I) Transcription factors dlx5a,dlx6a and osx. (J-L)
Parathyroid hormone pth1a and transcription factor runx2b.
doi:10.1371/journal.pone.0126928.g004
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
14 / 42


# Página 15

where pathways involving Rxra and its nuclear receptor dimerization partner Ppara were
down-regulated (S4 and S5 Tables). IPA comparison between PTH and VitD3 effects reveals
that, unlike the general metabolic effects exerted by VitD3, the most prominent biological func-
tions affected by PTH treatment were related to cell development, signaling and embryonic de-
velopment (S7 Table). The most highly developmentally affected systems were hematopoiesis
and the skeletal, muscular and cardiovascular systems. Further analysis revealed up-regulation
of a number of genes involved in or dependant on calcium metabolism, such as calreticulin
(CALR), integrin α9 (ITGA9), calcitonin receptor like (CALCRL) or arginine vasopressin re-
ceptor a1 (AVPR1A). Comparison of the genes affected by the two hormones exerting opposite
effects on bone formation, VitD3 and PTH, revealed only 12 genes in common (Fig 5A and
5B). Using these 12 genes allows building a regulatory network around the protooncogene
MYC and containing several genes that are differentially regulated in these two conditions (Fig
5A and 5C), suggesting opposing effects on mitochondrial (GSR), pyrimidine (CAD) or lipid
metabolism (CES1).
Effects of hypergravity on bone and general development
To compare the effects caused by bio-chemical or hormonal treatment on bone formation to
those exerted by bio-physical/mechanical constraints, we investigated the effects due to in-
creased gravity using the large diameter centrifuge (LDC) at the European Space Agency, ESA
(Noordwijk, Netherlands). In a first experiment, zebrafish larvae were grown at normal gravity
(1g) until 5dpf. One half of the population was brought to 3g hypergravity in the LDC for an-
other 4 days, while the other half was kept at 1g (see Fig 6). At 9dpf, the larvae were stained
with Alizarin red for bone matrix (Fig 7A and 7B) and analyzed as described above. No differ-
ence was observed between the two samples when total length of the larvae or size of the eye or
lens was determined (not shown). In the morphometric analysis, the 3g larvae present a larger
head skeleton with a significant increase of the distance between the 2 anguloarticular bones,
branchiostegal rays1, entopterygoid and the opercles (Fig 7C). In bone formation analysis (S3A
Table 1. Comparison of fold change values from the microarray dataset with those observed by RT-qPCR for VitD3 and PTH treatment.
VitD3
PTH
microarray
RT-PCR
microarray
RT-PCR
Gene
Fold Change
p-value
Fold Change
p-value
Fold Change
p-value
Fold Change
p-value
cad
1.424
0.094
2.017
< 0.001
cyp24a1
8.938
0.005
10.969
< 0.001
igfbp1
3.782
0.004
5.250
< 0.001
socs1
0.355
0.066
0.447
< 0.001
slc26a3
0.525
0.028
0.654
< 0.001
slc6a18
0.726
0.029
0.895
0.002
0.203
0.066
0.883
0.035
fgf4
0.520
0.110
0.777
< 0.001
0.450
0.079
0.831
< 0.001
mcph1
1.934
0.056
1.130
0.026
ndrg2
1.545
0.060
1.101
0.036
rxra
1.990
0.076
1.247
< 0.001
nrbp2
2.514
0.056
1.210
0.010
The fold change and statistical signiﬁcance (p-values) are given from the microarray data and the RT-qPCR conﬁrmation experiments. The data for the
genes selected for conﬁrmation of microarray results, respectively for VitD3 or PTH, are shaded in grey. slc6a18 and fgf4 were chosen for their regulation
by PTH and the results for VitD3 regulation are also shown
doi:10.1371/journal.pone.0126928.t001
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
15 / 42


# Página 16

Fig, S8 Table), the anguloarticular, branchiostegal ray2 and hyomandibular presented a clear
over ossification, while the ceratohyal presented a significantly higher proportion of advanced
ossification. In contrast, the dentary, maxilla and entopterygoid were not significantly affected
(S3A Fig). The global score obtained by addition of the scores of all the separate structures re-
vealed a significant increase of bone formation (from a score of 23± 4 to 27± 5.5) (Fig 7D). A clearly
weaker calcification was observed in the otoliths. More than 60% of the controls show 2 pairs of
dark otoliths (Fig 7A, 7B and 7E) compared to only lightly stained otoliths in the 3g group.
In addition, total mRNA was extracted from the larvae at 6dpf and whole genome gene ex-
pression was compared between larvae exposed for 1 day to 3g and 1g controls. The number of
genes found to be modulated by hypergravity was 499, although the extent of induction or
repression was surprisingly low (S9 Table), but significant as confirmed by RT-qPCR for 5 se-
lected genes (S10 Table). Interestingly, among the affected biological functions (S11 Table),
Fig 5. Comparison of genes affected after PTH or VitD3 treatment between 5–6dpf. (A) List of common genes and their respective log2(fold change) in
the two conditions. (B) Comparison of the number of genes affected by PTH or VitD3 treatment. The number of probes resulting in different hybridization
signals is given, with the numbers in parenthesis and the graph showing the numbers of IPA-annotated genes. (C) Network constructed using the common
genes and extended using the genes affected in one of the two conditions. The color overlay indicates the fold change after VitD3 (left) or PTH (right)
treatment. Genes up-regulated (red), down-regulated (green), (*) indicates that the gene is represented by two or more probes on the microarray.
doi:10.1371/journal.pone.0126928.g005
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
16 / 42


# Página 17

cellular and organism developmental processes ranked highest, only molecular transport appears
in second position. More specifically, development and function of the skeletal and muscular sys-
tem and connective tissue ranked highest, followed by the nervous and endocrine systems and fi-
nally hematological and cardiovascular systems. Among the specifically affected genes, many
transporter and ion channel genes are present, reminiscent of the observations after VitD3 treat-
ment. Interestingly, among the transcription factors, vitamin D receptor (vdr) is weakly, but sig-
nificantly down-regulated, similar to the nuclear receptor pparg. Other prominent transcription
factor genes are the homeo-box containing pou3f3 and its potential partners meis1 and onecut1.
Construction of specific networks in three different organ systems using IPA (S4 Fig) revealed
the inhibition of hubs like MYC, PPARG, vitamin D receptor (VDR), NFKBIA inhibitor in all sys-
tems, but also an extensive network specific to the cardiovascular system with, interestingly, a
down-regulation of the growth factor receptor/Ras mediator gene GRB2.
Effects of reduced gravity on bone and general development: "relative
microgravity"
As an approach to investigate some of the effects on zebrafish physiology to be expected when
going into real microgravity, we applied a protocol that we would qualify as "Reduced Gravity
Paradigm" or "relative microgravity". The principle is to grow the zebrafish larvae for a defined
Fig 6. Schematic overview of the different hypergravity experiments. (A) larvae are placed at hypergravity at 5dpf until 9dpf (3g), while (control) larvae
are kept at normal gravity for 9 days. Total mRNA was extracted at 6dpf and batches of larvae were fixed at 9dpf for Alizarin red staining of bone matrix. (B)
Experiment in which the control larvae were placed at 3g and kept at 3g until 6dpf (3g), or returned at 5dpf to 1g outside (3g>1g) or on the axis of the
centrifuge (3g>axe) for one day. An additional batch of larvae was kept at normal gravity until 6dpf (1g). RNA extraction and Alizarin red staining are
performed at 6dpf. For abbreviations see legend to Fig 1.
doi:10.1371/journal.pone.0126928.g006
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
17 / 42


# Página 18

Fig 7. Effect of 3g hypergravity between 5–9dpf on bone formation. (A,B) Alizarin red staining of 9dpf control larvae (A) and larvae treated for 4 days in
3g hypergravity after 5 days at 1g (B). Ventral view, anterior to the left. (C) Comparison of morphometric measurements for some selected distances within
the heads of control and 3g-treated larvae. Mean ± SD and t-test analysis were calculated for each measure on at least 20 individuals. * p < 0.05, ** p < 0.01
and ***p < 0.001. (D) Global score for bone formation in control and 3g treated larvae. (E) Comparison of cumulated frequencies of, respectively light, 1 pair
dark or two pairs dark otoliths in control and 3g treated larvae. For abbreviations see legend to Fig 1.
doi:10.1371/journal.pone.0126928.g007
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
18 / 42


# Página 19

period (5 days) in a hypergravity environment (this case 3g), before returning them to normal
gravity for one additional day (Fig 6B). The effect of this decrease in gravity on bone formation
and gene expression was then investigated.
Zebrafish fertilized eggs were subjected at 4hpf to 3g hypergravity until 5dpf. For compari-
son, a parallel batch was grown at normal gravity outside of the centrifuge chamber (1g). The
morphology of the embryos and larvae was monitored every day by microscopic observation,
no striking effect was observed on developmental processes such as segmentation, organogene-
sis or hatching time. Only a clearly decreased (delayed) pigmentation was observed at 24hpf
(Fig 8A and 8B), which was rapidly resolved as pigmentation was indistinguishable in 1g and
3g embryos at 2dpf.
At 5dpf, the larvae exposed for 5 days to 3g in the LDC, were separated in three distinct
batches, one was left in the LDC for another day (3g) while the other two were returned to nor-
mal gravity for one day. One batch was kept in a separate incubator outside of the centrifuge
chamber (3g>1g); the other was placed in an incubator positioned on the axis of the LDC
(3g>axe), in order to maintain a rotation movement without increasing the gravitational force.
The 1g batch continued to grow at normal gravity outside of the centrifuge chamber for the en-
tire 6 days.
At 6dpf, all larvae were collected and stained for calcified structures using Alizarin red.
Compared to larvae grown for 6 days at 1g, the bone structures in the head of all 3g exposed
larvae appeared more intense (Fig 8C–8F), more specifically the anguloarticular, maxillary
and, to a lesser extent the ceratohyal, hyomandibular and branchiostegal ray 1 (S12 Table). The
global score was significantly increased in all samples exposed to 3g for 5 or 6 days (Fig 8G).
Morphological analysis revealed a significant increase in the distance between branchiostegal
rays 1, enteropterygoids and opercles, and an increase in the parasphenoid area (Fig 9).
A central gene network is rapidly activated in reduced gravity
At 6dpf, all larvae were collected and used for mRNA extraction. Gene expression was deter-
mined by micro-array analysis, larvae exposed to 3g for the entire 6 days were chosen as con-
trol (S13–S15 Tables). Relative to this hypergravity sample, a remarkable similarity was
observed in the biological functions affected in the normal gravity larvae (Table 2). Among the
top ten functions modulated in each condition we found, on the one hand cell growth and pro-
liferation, development, death and survival, organization and function, on the other hand em-
bryonic and organismal (organ) development with a focus on connective tissue and
cardiovascular development in the 6 days control at 1g. Only 3g>axe larvae presented 7 affect-
ed genes related to "auditory and vestibular system", related to their stay on a purely
rotating position.
When comparing the affected genes in the three conditions, it appears that 16 genes are
common to all three (Fig 10), while 20 genes are common only to the 1g samples between days
5 and 6 (3g>1g and 3g>axe). Respectively, 69 and 20 genes are common between the static 1g
for 1 day (3g>1g) or rotating 1g (3g>axe) for 1 day and the larvae having spent all 6 days at 1g
(1g). Several genes, mostly common to all three conditions, were selected and the modulation
of their expression was confirmed by RT-qPCR (S16 Table). Regulatory networks were con-
structed using the genes common to all three conditions, but also using those common to the
1g for one day condition (3g>1g and 3g>1axe) (Fig 11). Strikingly, a network composed of 7
genes (FOS, FOSB, EGR1, EDN1, SOCS3, GADD45B, KLF2) that were affected in exactly the
same manner in all three conditions could be constructed, indicating that they represent a cen-
tral network that is affected by gravitational conditions. Most importantly, these central genes
were affected to the same extent, relative to the 3g for 6 days control, whether the larvae were
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
19 / 42


# Página 20

Fig 8. Effect of "relative microgravity" between 5–6dpf on bone formation. (A, B) comparison of
pigmentation at 24hpf in 1g (A) and 3g (B) larvae. (C-F) Alizarin red staining of larvae kept at 1g until 6dpf (1g,
C), control larvae kept at 3g until 6dpf (3g, D), larvae kept at 3g until 5dpf and returned to 1g off the centrifuge
(3g>1g, E) or on the axis (3g>axe, F), Ventral view, anterior to the left. (G) Global scores for bone formation in
control and the different treated larvae.
doi:10.1371/journal.pone.0126928.g008
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
20 / 42


# Página 21

kept at 1g during the entire experiment or only for the last day, suggesting that their expression
levels are specific to this gravitational condition and are rapidly (within one day) adapted to
new conditions. Five additional genes (MVP, HBE1, HES5, SOX10, LGALS3BP) were only af-
fected after 1 day at lower gravity (both 3g>1g and 3g>1axe), indicating that they may be actu-
ally involved in the mechanism for rapid adaptation to lower gravity. Further analyses were
performed using all the genes common to any two of the conditions (S5 Fig), also analyzed ac-
cording to their potential function in individual organ systems (S6 Fig). By extending the net-
work that way, other nodes become apparent, such as the nuclear receptor PPARG, the protein
chaperone HSP90AA1 and the regulatory peptide endothelin (EDN1) (S5 Fig). Expression of
NFKBIA, a target gene for the NFkB pathway coding for an inhibitor of this pathway, was de-
creased in two conditions, potentially causing the decreased expression of the antiproliferative
factor BTG2 [80] observed in all three conditions.
Finally, we compared the genes affected in the 1g>3g experiment, which experienced a shift
from 1g to 3g on day 5, with those affected in the 3g>1g experiment where the larvae were re-
turned to 1g after 5 days at 3g. Among the affected genes, 41 were common to both
Fig 9. Morphometric analysis of bone elements at 6dpf after "relative microgravity”. The distances are measured in pixels. Mean ± SD and t-test
analysis were calculated for each measure on at least 20 individuals. (A) Distances between the different cranial bone elements. (B) Area of the
parasphenoid bone. * p < 0.05 and ***p < 0.001. For abbreviations see legend to Fig 1.
doi:10.1371/journal.pone.0126928.g009
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
21 / 42


# Página 22

Table 2. Biological functions associated to "relative microgravity"-affected genes.
1g
3g > axe
3g > 1g
Category
p-value
N
p-value
N
p-value
N
Cellular Growth and Proliferation
5.71E-11-4.98E-03
181
6.87E-07-6.95E-03
70
5.78E-09-3.15E-03
213
Cell Cycle
1.15E-10-4.98E-03
93
3.61E-05-7.6E-03
20
8.79E-06-2.63E-03
85
Organismal Survival
8.1E-09-8.1E-09
119
1.23E-04-7.24E-03
46
1.87E-06-3.23E-03
139
Cellular Development
4.9E-08-4.98E-03
156
6.89E-08-8.09E-03
60
2.17E-06-3.31E-03
205
Connective Tissue Development and Function
4.9E-08-4.98E-03
44
1.55E-04-6.95E-03
26
1.91E-05-3.3E-03
75
Tissue Development
4.9E-08-4.6E-03
104
6.89E-08-8.11E-03
65
2.26E-06-3.31E-03
177
Cell Death and Survival
1.79E-07-4.99E-03
157
6.01E-09-8.25E-03
59
1.49E-10-3.35E-03
203
DNA Replication. Recombination. and Repair
1.11E-06-4.98E-03
77
4.27E-03-7.37E-03
7
2.22E-03-2.63E-03
8
Cardiovascular System Developt and Function
9.78E-06-3.03E-03
28
1.1E-05-7.81E-03
28
3.76E-06-3.02E-03
92
Hematological System Developt and Function
9.78E-06-4.98E-03
57
4.53E-05-8.11E-03
32
6.3E-06-3.31E-03
114
Cellular Assembly and Organization
1.75E-05-4.98E-03
80
4.21E-06-7.65E-03
42
1.83E-05-3.02E-03
90
Cellular Movement
2.64E-05-4.55E-03
94
2.43E-04-7.89E-03
35
1.82E-06-3.02E-03
132
Cell Morphology
3.38E-05-3.03E-03
89
2.43E-04-7.65E-03
47
7.12E-08-2.7E-03
144
Amino Acid Metabolism
4.27E-05-4.98E-03
23
2.37E-03-6.69E-03
3
1.08E-05-1.83E-03
13
Small Molecule Biochemistry
4.27E-05-4.98E-03
100
1.6E-05-7.6E-03
36
1.08E-05-2.74E-03
72
Embryonic Development
4.66E-05-4.98E-03
83
6.89E-08-7.81E-03
52
2.46E-07-3.22E-03
141
Organismal Development
4.66E-05-4.93E-03
85
6.89E-08-7.81E-03
63
2.46E-07-3.22E-03
203
Cell-To-Cell Signaling and Interaction
7.89E-05-4.39E-03
27
6.44E-05-7.6E-03
17
2.07E-03-2.07E-03
7
Cellular Function and Maintenance
1.27E-04-4.39E-03
67
4.21E-06-7.65E-03
37
1.83E-05-3.02E-03
148
Energy Production
1.72E-04-2.24E-03
22
5.95E-03-7.37E-03
6
Lipid Metabolism
1.72E-04-4.98E-03
60
6.84E-05-7.6E-03
30
4.52E-05-2.74E-03
58
Renal and Urological Developt and Function
1.72E-04-2.93E-03
25
2.04E-04-7.6E-03
5
2.56E-03-2.56E-03
2
Nucleic Acid Metabolism
1.74E-04-3.03E-03
36
1.6E-05-7.37E-03
15
2.63E-03-2.63E-03
3
Tissue Morphology
1.75E-04-4.98E-03
80
4.53E-05-7.81E-03
47
2.49E-06-3.15E-03
129
Cellular Compromise
2.25E-04-9.79E-04
13
5.37E-04-7.6E-03
15
4.79E-04-9.19E-04
9
Molecular Transport
2.25E-04-4.98E-03
89
1.6E-05-7.6E-03
41
4.52E-05-2.74E-03
106
Lymphoid Tissue Structure and Developt
2.29E-04-4.55E-03
24
1.18E-03-6.38E-03
17
2.24E-05-3.01E-03
36
Gene Expression
4.45E-04-4.39E-03
83
2.04E-04-7.81E-03
37
6.97E-08-2.28E-03
134
Carbohydrate Metabolism
5.2E-04-3.1E-03
43
6.84E-05-7.52E-03
19
2.63E-03-2.63E-03
3
Hematopoiesis
5.21E-04-4.55E-03
10
2.78E-03-7.76E-03
14
6.3E-06-3.31E-03
70
Hair and Skin Development and Function
6.67E-04-4.49E-03
19
1.88E-03-1.88E-03
6
8.79E-06-3.35E-03
47
Nervous System Developt and Function
8.28E-04-4.98E-03
40
8.72E-06-7.6E-03
46
2.37E-05-2.7E-03
76
Organ Morphology
8.73E-04-3.03E-03
18
3.32E-05-6.97E-03
33
4.58E-06-3.11E-03
87
Organ Development
1.06E-03-4.24E-03
33
1.25E-06-7.6E-03
35
2.26E-06-3.01E-03
109
Skeletal and Muscular Developt and Function
1.06E-03-4.28E-03
34
1.55E-04-7.6E-03
23
2.26E-06-3.11E-03
67
Immune Cell Trafﬁcking
1.54E-03-1.71E-03
3
8.11E-03-8.11E-03
5
2.23E-04-2.79E-03
53
Reproductive System Developt and Function
1.54E-03-4.6E-03
10
2.37E-03-7.6E-03
8
1.21E-03-2.81E-03
21
Visual System Development and Function
1.7E-03-1.71E-03
6
1.25E-06-4.5E-03
17
4.58E-06-1.78E-03
35
Post-Translational Modiﬁcation
1.71E-03-4.98E-03
4
8.8E-05-5.83E-03
15
3.03E-04-1.43E-03
48
Digestive System Development and Function
2.64E-03-4.24E-03
8
1.17E-04-8.04E-03
24
6.28E-05-1.79E-03
52
Hepatic System Development and Function
2.64E-03-4.24E-03
6
2.04E-04-6.95E-03
15
1.02E-04-7.51E-04
28
Protein Synthesis
2.98E-03-2.98E-03
41
8.8E-05-3.54E-03
27
2.6E-04-2.24E-03
66
Vitamin and Mineral Metabolism
3.55E-03-3.55E-03
8
1E-03-3.18E-03
6
2.56E-03-2.56E-03
2
Organismal Functions
4.98E-03-4.98E-03
2
6.73E-04-4.27E-03
8
3.15E-03-3.15E-03
13
Protein Trafﬁcking
4.98E-03-4.98E-03
2
2.45E-03-2.45E-03
19
Cell Signaling
8.8E-05-5.02E-03
14
6.4E-04-1.05E-03
46
(Continued)
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
22 / 42


# Página 23

experiments (Fig 10) that could be assembled in a regulatory network (S7 Fig). Two regulatory
genes attracted our attention due to their increased expression in the 3g environment (note the
fold change relative to the 1g control in the 1g>3g, and relative to the 3g sample in the 3g>1g
experiment): SOX3 is a transcription factor shown to be involved in neural, pituitary and cra-
niofacial development [81], while the HEY1 gene is a target of Notch signaling and was shown
to regulate bone homeostasis [82]. Two other genes, coding for embryonic hemoglobin HBE1
and the oligopeptide transporter SLC15A1 were down-regulated at 3g.
Discussion
Zebrafish present remarkable degrees of similarity with mammals in the molecular mecha-
nisms involved in their developmental biology and physiology. Moreover, their ease of hus-
bandry, high fecundity, and small size paves the way for a possible future space experiment,
triggering the proposal of their use for the study of gravitational biology [83–89]. We decided
to explore the effects of increased gravity (hyper-g) on zebrafish larvae using the large diameter
centrifuge (LDC). This device allows applying a well-controlled and constant centrifugal force
(1g-20g) by minimizing, through the large diameter of the rotating arms, the possible effects of
Coriolis force [64].
Our aim was to concentrate on the effects on bone formation, therefore we chose to start the
experiments at 5dpf, when perichondral ossification is taking place within all major cranial car-
tilage elements and intramembranous bone formation is ongoing. We evaluated the effects on
cartilage and bone formation by staining these structures after several days of treatment, at 9 or
10dpf. For a more detailed, more accurate and more objective evaluation of skeletal develop-
ment, we developed two different, but complimentary methods for analyzing images of stained
zebrafish larvae. The first one uses a number of landmarks placed manually within the images
(using the software environment CYTOMINE) and allows automatic extraction of distances
and angles between these landmarks, ultimately resulting in a morphometric description of the
head skeleton. The second one is based on manually assigning a developmental score to each
cranial bone element within each image, enabling us to calculate a mean score for each element
and a global score for each individual.
To validate these approaches, we performed two treatments of zebrafish larvae whose effects
had been previously described [63]. The first treatment uses exogenous vitamin D3 (VitD3)
[90] to increase bone formation, indeed the general VitD3 metabolism in teleosts is similar to
that in mammals, teleosts possess two vitamin D receptors (VDRs) and knock-down of VDRa
expression causes a decrease of calcium ion uptake [90]. PTH and related peptides are known
hypercalcemic agents in mammals, however their function is more controversial in teleosts, de-
pending on the species [91]. Although teleosts do not present a parathyroid gland, they do
Table 2. (Continued)
1g
3g > axe
3g > 1g
Category
p-value
N
p-value
N
p-value
N
Protein Degradation
3.54E-03-3.54E-03
13
2.24E-03-2.24E-03
13
Behavior
3.61E-04-3.82E-03
22
5.52E-06-3.15E-03
77
Auditory and Vestibular System Developt and Function
5.5E-04-3.82E-03
7
Ingenuity Pathway Analysis of the lists of genes affected at 6dpf after 6 days at 1g (1g), or after 5 days at 3g and returned to 1g on the centrifuge axis
(3g>axe) or outside of the centrifuge room (3g>1g), each time compared to 3g hypergravity treatment for 6 days (3g). Columns indicate respectively the
function, the range of p-values (signiﬁcance) associated to various sub-functions, and the number of genes concerned (N).
doi:10.1371/journal.pone.0126928.t002
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
23 / 42


# Página 24

produce PTH in the gills, probably in cells identified by the expression of gcm2, a gene whose
orthologues are required for parathyroid development in chicken and mouse [92, 93]. PTH ad-
ministration induced hypercalcemia in fugu (Tetraodon nigrividans) by inducing both osteo-
blast and osteoclast function and by decreasing scale calcium content [94]. Genes homologous
to the mammalian PTH-related peptides (PTHrP) were found in teleosts, they are more widely
expressed [95], they increase calcium uptake in sturgeon (Acipenser nacarii) [96]and were
shown to play different roles in craniofacial development in zebrafish [97]. Blocking PTH sig-
naling through the use of a PTH/PTHrP antagonist resulted in a decreased hypercalcemic
Fig 10. Number of genes affected in the various hypergravity experiments. The absolute number of
probes resulting in a statistically significant hybridization signal is given for each condition. In parentheses,
the corresponding number of genes with an annotation in IPA is given, while the Venn diagrams represent the
number of genes unique to each condition and genes common to two or three conditions.
doi:10.1371/journal.pone.0126928.g010
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
24 / 42


# Página 25

Fig 11. Network of genes affected in "relative microgravity" experiments. A network was constructed
using the genes common to all three experiments, or the genes common only to 3g>1g and 3g>axe. Color
overlay indicates the fold change relative to the 3g sample taken as control. Genes up-regulated (red), down-
regulated (green), (*) indicates that the gene is represented by two or more probes on the microarray.
doi:10.1371/journal.pone.0126928.g011
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
25 / 42


# Página 26

response to estradiol in sea bream (Sparus aurata) [98]. Finally, four stanniocalcin (stc) genes
are present in fugu and zebrafish, only stc1-a expression was sensitive to the calcium concen-
tration in water [99].[98] while PTHrP and Stc were shown to have opposing effects on calcium
uptake in intestinal explants [100]. Depending on the mode of administration (intermittent or
continuous) PTH and PTHrP were shown, respectively to increase or decrease bone formation
in zebrafish [101] or seabream [102].
We confirm the effects described in zebrafish on general bone formation and in addition,
the combined approach allowed us give a more detailed description of these effects. Although
the general morphology was preserved in both cases, VitD3 treatment lead to a broader jaw
both in cartilage and bone and a longer head in bone, while PTH treatment leads to an in-
creased length of the ceratohyal cartilage, a general decrease of ossification, a decreased length
of the parasphenoid bone and a broadening of the posterior head skeleton. The discrepancy be-
tween cartilage and bone concerning the longer head probably results from the fact that the
landmarks used in bone (parasphenoid and notochord) do not have a real equivalent in carti-
lage and may mineralize independently from it. When we applied the same method to larvae
subjected to hypergravity, we observed a broadening of the entire head skeleton (increased dis-
tance between symmetrically paired elements), for both types of treatment: 3g between 5–9dpf
(1g>3g experiment, Fig 7A–7C), and 3g between 0 and 6dpf (experiments 3g, 3g>1g and
3g>axe, Fig 8). Similarly, the developmental scoring method allowed a more differentiated de-
scription of the observed effects (Fig 12). While VitD3 treatment caused a generally significant
increase in ossification of most elements, this was less prominent for the maxillary and absent
for the anguloarticular. Conversely, the decrease of ossification caused by PTH treatment was
significant for all elements except branchiostegal rays 1. Increased ossification was significant
only in the anguloarticular and ceratohyal after 3g treatment between 5–9dpf (1g>3g), but ex-
tended to the maxillary in the earlier treatments from 0–5 or 6dpf. Importantly, exposing the
larvae for 6 days to 3g (3g condition) or returning them to 1g for the last day (3g>1g and
3g>axe) did not significantly affect bone formation, indicating that 1 day of altered gravity is
not sufficient to cause morphological changes in the skeleton. Understanding of the molecular
mechanisms underlying these differential effects on the various skeletal elements and their
morphology will require further investigation.
Exposure to 3g starting at 5dpf (1g>3g condition) led to increased bone calcification in the
anguloarticular and ceratohyals at 9dpf (Fig 7), while the otoliths were clearly less stained. The
decrease in otolith calcification was already previously described [103, 104] and was proposed
to involve a regulatory mechanism linking gravity sensing to the production of carbonic anhy-
drase and other matrix proteins in the inner ear [105–107]. Thus, the decrease in otolith calcifi-
cation after prolonged exposure to 3g was expected, but it also emphasizes the specificity of the
observed increase in ossification.
During early exposure to 3g (in the "relative microgravity" experiments), we observed a
transient delay in pigmentation at 24hpf, which was rapidly resorbed at 48hpf. This finding is
reminiscent of the transient decrease in the number of melanocytes that was observed at 24hpf
during early exposure to simulated microgravity using a Rotating Wall Vessel device [51]. It is
at present unclear whether a common mechanism may explain such a similar delay both in
hypergravity and in simulated microgravity.
We then turned to studying differences in gene expression caused by the various treatments.
We chose to perform these studies using mRNA from entire larvae, as methods for isolation of
specific cells, such as dissection or fluorescent cell sorting might not be available in a future
space experiment. First, we followed expression of bone-specific genes during normal develop-
ment between 6 and 10dpf. We observed a sharp rise of mRNA coding for bone matrix proteins
Sparc, Bglap, Spp1 and Col1a2 followed by a rapid decrease after 7dpf, suggesting that the
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
26 / 42


# Página 27

major part of the bone matrix is formed at 7dpf and that further ossification is mainly due to
mineral deposition. This is consistent with the observed sharp decrease of osx expression, fol-
lowed with some delay by dlx5a expression, both indicating a decrease in osteoblast differentia-
tion. The continuous decrease in the levels of col10a1a mRNA could be related to the proposed
inhibitory effect of this factor on biomineralization [108, 109], while the large increase of
runx2b and pth1a mRNA during the entire period could be related to some other functions of
these factors [110, 111]. Following the modulation of gene expression during chemical treat-
ments revealed a clear upward trend for bone matrix protein-encoding genes upon VitD3 treat-
ment and a clear downward trend during PTH treatment. These trends are consistent with the
assumption that bone matrix secretion plays a functional role in the observed increase or de-
crease, respectively, in bone formation. Expression of osx is increased during the first day of
VitD3 treatment and decreased during PTH treatment, again consistent with a respectively
prolonged or shortened period of osteoblast differentiation, also further supported by the in-
crease of dlx5a and dlx6a expression at 8–9dpf during VitD3 treatment.
Fig 12. Summary graphs comparing the bone formation scores for each structure in the different experiments. Statistical analysis was performed by
X² of Pearson and a logistic regression. In red, the scores are significantly increased. In green, the scores are significantly decreased. (A) PTH. (B) VitD3. (C)
3g hypergravity between 5–6dpf (D) "relative microgravity". For abbreviations see legend to Fig 1.
doi:10.1371/journal.pone.0126928.g012
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
27 / 42


# Página 28

Finally, to determine the effects on whole genome gene expression of the various treatments,
we chose to concentrate on mRNA levels only after one day of treatment, as we are mainly in-
terested in regulatory events. A summary of all the genes affected by any of the studied condi-
tions is shown in S17 Table. Again, we validated our approach by investigating the effects of
known regulators of bone formation. As expected, VitD3 treatment induced cyp24a1 expres-
sion, while PTH administration led to a decrease in endogenous pth1a expression. Further-
more, VitD3 treatment caused significant changes in overall metabolism, as shown by the
involvement of affected genes in molecular transport or lipid metabolism. Probably for this rea-
son, functions related to embryogenesis or organ morphology rank much lower in the list of af-
fected pathways. These findings are consistent with previous results, obtained using a deep
sequencing (RNA-seq) approach, which also showed a high proportion of metabolic pathways
affected by VitD3 treatment, administered either between 2 and 6–7dpf or between 6–7dpf
[112]. In contrast, PTH treatment affected less genes, but these were more involved in develop-
mental processes. Interestingly, several genes were regulated in opposite directions upon VitD3
or PTH treatment (Fig 5A and 5C), suggesting that they may be involved in the opposite effects
on bone mineralization that we observed. However, when we classified the genes according to
their known involvement in specific organ function (S4 Fig), these genes were more specifically
known for their function in muscle, indicating that further investigations are required.
When comparing genes and pathways affected by hypergravity, cellular growth and prolif-
eration functions ranked very high, followed by cellular, tissue and organismal development
(Table 2). Among the canonical pathways affected (S18 Table), we found those involving IGF,
as already mentioned, and those involving pituitary hormones Prl and Gh as well as nuclear re-
ceptors. Interestingly, finer analysis of the affected biological functions revealed that all hyper-
gravity conditions acted on organism survival and cell apoptosis (S19 Table), although no
effect on larval survival or growth was observed in our experiments. Affected regulatory net-
works comprise PPARG, involved in adipocyte differentiation and regulating blood glucose
uptake, consistent with the presence of other genes connected to insulin function. This obser-
vation may be related to previous experiments in rodents that showed a decrease in fat mass in
hypergravity [113, 114]. Another gene consistently induced by hypergravity in mammals is the
Hsp70 stress response gene [113, 114]. In zebrafish kept for the first two days at 3g, increased
expression of a fluorescent reporter transgene hsp70-gfp hypergravity was shown mainly in the
lens [49], however no induction of the hsp70 gene was observed here, probably due to the later
observation stages. This indicates that older fish larvae are probably less stressed by hypergrav-
ity than are mammalian systems. Note that changes in the fli1-gfp transgene expression were
also only observed for exposures before 24hpf [115]. Similarly, a decrease of ß-actin-gfp trans-
gene expression was described in Rohon-Beard neurons [115], which disappear after 80hpf.
Other important nodes are the NFKBIA inhibitor of the NF-kB pathway, involved in immune
and inflammatory responses, and the multifunctional MYC gene.
The c-FOS gene was first described as the cellular homolog of the viral oncogene causing
murine osteosarcoma [116], while gene knock-out mice suffered from severe defects in bone
development and haematopoiesis [117]. First microgravity experiments in murine carcinoma
cells revealed a decreased induction of c-Fos and its heterodimeric partner c-Jun by growth fac-
tors [118, 119]. Decreased c-Fos expression in microgravity was also observed in osteoblastic
cells [120, 121], while exposure to intense hypergravity (50–90g) caused an increased expres-
sion of c-Fos and Egr1 [122]. More moderate hypergravity conditions (3g) also revealed rapid
(36 min) induction of c-Fos expression in osteoblasts [123], while both hypergravity loading
and unloading caused increased expression in rat brains [124, 125]. This latter c-Fos induction
was then considered as an indicator for neural activity in specific brain regions, in particular
those related to vestibular sensing and processing [126–128]. Here, we show that exposure of
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
28 / 42


# Página 29

zebrafish embryos to 3g hypergravity during the first 5–6 days of development leads to in-
creased expression of fos, as part of a regulatory network composed of 6 other genes (fosb, egr1,
edn1, socs3a, gadd45b, klf2a) that are induced in 3g conditions. Among these, the fos homolog
fosb and the Zn-finger transcription factor gene egr1 belong to the immediate-early class of
genes that are rapidly induced by growth factors. In mouse, FosB knock-out leads to behavioral
defects [129], while Egr1 null mice display sterility, impaired growth and pituitary develop-
ment [130, 131]. Egr1 was also rapidly induced in osteoblast cells upon mechanical stress
[132]. In zebrafish [133], egr1 was shown to be part of a regulatory cascade controlling cartilage
development [134] that is induced by Fgf signaling [135]. Edn1 is a vasoconstrictor peptide
whose absence causes elevated blood pressure and craniofacial abnormalities [136] in mouse,
while a zebrafish edn1 mutant displayed mainly defects in cranial cartilage development [7,
137]. Socs3 is a suppressor of cytokine signaling; in mouse it was shown to inhibit placental
and fetal liver erythropoiesis [138], while a zebrafish mutant in the paralog socs3a was deficient
in hair cell development and regeneration in the inner ear and the lateral line neuromasts
[139]. Gadd45b is a factor causing growth arrest upon DNA-damage, but also involved in he-
matopoiesis and immune response [140]. Finally, loss of the Klf2 gene in mouse causes defects
in vascular, skeletal and craniofacial development and in erythropoiesis [141], while a zebrafish
klf2a mutant displayed impaired cardiac valve development due to a deficient response to
blood flow [142]. Klf2a was further shown to be required for nitric oxyde (NO) synthesis dur-
ing artery and hematopoietic stem cell development [143], a process that is also highly involved
in bone development [144–146]. Taken together, the network formed by these seven genes that
are up-regulated in 3g conditions carries the potential to affect most processes that are known
to be influenced by gravitational changes; from vestibular gravity sensing to hematopoiesis, im-
mune response, vascular system and finally the skeletal system as was illustrated here. More-
over, this network is activated not only in larvae grown at 3g relative to larvae grown at 1g for 6
days, but also relative to larvae grown at 3g for 5 days and then returned to 1g for only one day
(Figs 10 and 11). Increased expression of this gene network appears to be specific for hyper-
gravity, while expression rapidly returns to normal after 1 day at 1g.
Five genes could be connected to this regulatory network that were specifically up-regulated
(MVP, HBE1, SOX10, LGALS3BP) or down-regulated (HES5) after return to 1g conditions for
1 day (Fig 11). In mouse, Sox10 knock-out leads to neurological defects [147], while sox10 mu-
tant zebrafish are deficient in melanocyte pigmentation and inner ear development [148–150].
Similarly, Hes5 was shown to regulate neurogenesis [151], but also human cartilage differentia-
tion under the control of Notch signaling [152]. Lgals3bp was shown to play a role in immune
response and cell adhesion [153]. HBE1 codes for one of the embryonic hemoglobins, suggest-
ing alterations in oxygen transport under different gravity conditions. MVP is a component of
the ribonucleoprotein "vault" structures involved in nucleo-cytoplasmic transport and signal
transduction [154]. Interestingly, loss of function studies for Mvp in zebrafish revealed defects
in brain development and the response to mechanical stimulus (touch) [155]. The precise role
of these genes in detection of decreased gravity and signal transmission to other physiological
systems remains to be established.
Comparison of the 1g>3g and the 3g>1g experiments revealed the increased expression in
hypergravity of two regulatory genes, SOX3 and HEY1, which both may play a role in bone de-
velopment and/or homeostasis [81, 82], while HBE1 and SLC15A1 were down-regulated at 3g.
Interestingly, only HBE1 is also regulated in the 3g>axe experiment, further supporting a gen-
eral effect on oxygen transport, while only GADD45B expression was affected in all 3g experi-
ments. None of the other genes composing the common regulatory network in "relative
microgravity" was affected in the 1g>3g experiment. Actually, the overall effect of 1 day expo-
sure to 3g was surprisingly small at the genome level, compared to the other hypergravity
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
29 / 42


# Página 30

experiments (Tables S11, S13–S15), a result that is reminiscent of that observed previously in
mammalian renal cells [156]. This observation suggests that the "Reduced Gravity Paradigm" is
not simply a reversed hypergravity experiment, but rather that it represents a specific experi-
mental condition. Future experiments will reveal whether this approach may be considered as
a good approximation of microgravity.
In conclusion, we present an approach to objectively characterize cranial skeletal develop-
ment in zebrafish larvae by morphometric image analysis and used this method to further
characterize the effects of VitD3 and PTH on cartilage and bone formation. We have followed
the expression of selected bone-related genes during 5 days of VitD3 or PTH treatment and an-
alyzed whole genome gene expression after 1-day treatment. We have compared and correlated
these results to the effects of hypergravity exposure on cranial skeleton formation. Finally, we
have implemented a new type of hypergravity experiment, the "Reduced Gravity Paradigm",
which allowed identification of a regulatory network of seven genes that are up-regulated in 3g,
as well as several genes whose expression is rapidly modified when switching between 1g and
3g regimes. Future investigations will reveal whether these gene regulations are specific for par-
ticular organ systems and how they contribute to the overall physiological adaptation to altered
gravitational environments.
Supporting Information
S1 Fig. Morphometric analysis of cartilage staining after 5 days chemical treatments. The
distances are measured in pixels. Mean ± SD and t-test analysis were calculated for each mea-
sure on at least 20 individuals.  p < 0.05 and p < 0.001. (A, C) Distance after VitD3 treat-
ment. (B, D) Distance after PTH treatment. Abbreviations as in 1. A) Morphometric analysis
in VitD3-treated larvae cartilage revealed an increase of the distance between articulation (ar)
"up" and "down", leading to a broader jaw as compared to untreated animals, while (A, C) all
the other distances remained unchanged. B) Morphometric cartilage analysis of larvae treated
with PTH for 5 days revealed a significant increase in length of the ceratohyal cartilages only
(D).
(JPG)
S2 Fig. Gene pathways affected after VitD3 treatment between 5–6dpf. Genes filtered ac-
cording to the described function for their human homologs using IPA in muscle or bone func-
tion. Genes up-regulated (red), down-regulated (green), () indicates that the gene is
represented by two or more probes on the microarray.
(JPG)
S3 Fig. Changes in the extent of bone formation in hypergravity experiments. Bone devel-
opment is classified for each element into different categories: Absent (no structure present;
red), early ossification (beginning of the bone ossification; yellow), advanced ossification (the
structure is present and already developed as the control; green) and over ossification (the
structure is more developed compared to the control; purple). Cumulated frequencies in % are
represented for each element. As no significant difference was observed for paired structures
between left and right (up and down), their scores have been combined. Statistical analysis was
performed by X² of Pearson and a logistic regression. (A) Cumulated frequency after 3g be-
tween 5–9dpf. (B) Cumulated frequency at 6dpf in the larvae left for 6 days at 3g, or the "rela-
tive microgravity" experiments (3g-axe and 3g>1g) relative to the 1g control. For abbreviations
see legend to 1.
(JPG)
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
30 / 42


# Página 31

S4 Fig. Regulatory networks related to different tissues after 3g hypergravity between
5–6dpf. Genes filtered according to the described function for their human homologs using
IPA in bone, muscle, or cardiovascular system function. Genes up-regulated (red), down-
regulated (green), () indicates that the gene is represented by two or more probes on the mi-
croarray.
(JPG)
S5 Fig. Network of genes affected in "relative microgravity" experiments. A network was
constructed using the genes common to any two of the three experiments. The color overlay in-
dicates the fold change in each experiment (1g, 3g>1g and 3g>axe) relative to the 3g sample
taken as control. Genes up-regulated (red), down-regulated (green), () indicates that the gene
is represented by two or more probes on the microarray.
(JPG)
S6 Fig. Tissue-specific networks of genes affected in "relative microgravity" experiments.
Networks were constructed using the genes common to any two of the three experiments and
filtered according to the described function for their human homologs using IPA in bone,
muscle or cardiovascular system function. The color overlay indicates the fold change in the 1g
experiment (1g, 3g>1g and 3g>axe) relative to the 3g sample taken as control. Genes up-regu-
lated (red), down-regulated (green), () indicates that the gene is represented by two or more
probes on the microarray.
(JPG)
S7 Fig. Network of genes affected in "relative microgravity" and 3g between 5–6dpf
(1g>3g) experiments. A network was constructed using the genes common to the 3g>1g and
1g>3g experiments. The color overlay indicates the fold change in each experiment relative to
the respective control: control is 1g for the 1g>3g, and 3g for the 3g>1g experiment. Genes
up-regulated (red), down-regulated (green), () indicates that the gene is represented by two or
more probes on the microarray.
(JPG)
S1 Table. List of oligonucleotides used for RT-qPCR experiments.
(DOCX)
S2 Table. Ossification scores for individual bone elements in control and 5 days VitD3--
treated larvae. (A) The bone structures distributed in 2 categories (early and advanced ossifica-
tion) (B) The bone structures distributed in 3 categories (early, advanced and over ossification)
(DOCX)
S3 Table. Ossification scores for individual bone elements in control and 5 days PTH-treat-
ed larvae. (A) The bone structures distributed in 2 categories (early and advanced ossification)
(B) The bone structures distributed in 3 categories (absent, early and advanced ossification)
(DOCX)
S4 Table. Genes affected in larvae treated with VitD3 between 5–6dpf relative to control.
The indicates the human homolog of the gene, its "Entrez" gene name, the log ratio of VitD3--
treated larvae compared to control, the presence of duplicate probes on the microarray (D) and
the type of protein it encodes. Genes are arranged according to their type and in alphabetical
order.
(DOCX)
S5 Table. Biological functions associated to genes affected by VitD3. Ingenuity Pathway
Analysis of the list of genes affected at 6dpf after VitD3 treatment for 24 hours. Columns
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
31 / 42


# Página 32

indicate respectively the function, the range of p-values (significance) associated to various
sub-functions, and the number of genes concerned (N).
(DOCX)
S6 Table. Genes affected in larvae treated with PTH between 5–6dpf relative to control. The
table indicates the human homolog of the gene, its "Entrez" gene name, the log ratio of PTH-
treated larvae compared to control, the presence of duplicate probes on the microarray (D) and
the type of protein it encodes. Genes are arranged according to their type and in alphabetical
order.
(DOCX)
S7 Table. Biological functions associated to genes affected by PTH. Ingenuity Pathway Anal-
ysis of the list of genes affected at 6dpf after PTH treatment for 24 hours. Columns indicate re-
spectively the function, the range of p-values (significance) associated to various sub-functions,
and the number of genes concerned (N).
(DOCX)
S8 Table. Ossification scores for individual bone elements in control and 3g-treated larvae
between days 5–6dpf. The fraction (in %) of larvae presenting the indicated score for each ele-
ment is given, together with the statistical evaluation of a significant difference compared to
control. (A) The bone structures distributed in 2 categories (early and advanced ossification)
(B) The bone structures distributed in 3 categories (absent, early and advanced ossification)
(DOCX)
S9 Table. Genes affected in larvae placed at 3g between 5 and 6dpf (1g>3g) relative to con-
trol. The indicates the human homolog of the gene, its "Entrez" gene name, the log ratio com-
pared to larvae kept at 1g between 0 and 6dpf, the presence of duplicate probes on the
microarray (D) and the type of protein it encodes. Genes are arranged according to their type
and in alphabetical order.
(DOC)
S10 Table. Comparison of fold change values from the microarray dataset with those ob-
served by RT-qPCR for larvae placed at 3g between 5 and 6dpf (1g>3g) relative to control.
The fold change and statistical significance (p-values) are given from the microarray data and
the RT-qPCR confirmation experiments.
(DOCX)
S11 Table. Biological functions associated to genes affected by hypergravity between
5–6dpf (1g>3g). Ingenuity Pathway Analysis of the list of genes affected at 6dpf after 3g
hypergravity treatment for 24 hours (1g>3g). Columns indicate respectively the category of
function, the range of p-values (significance) associated to various sub-functions, and the num-
ber of genes concerned.
(DOCX)
S12 Table. Ossification scores for individual bone elements in larvae placed at 1g or 3g for
6 days or returned to 1g the last day. The fraction (in %) of larvae presenting the indicated
score for each element is given, together with the statistical evaluation of a significant difference
compared to control. (A) The bone structures distributed in 2 categories (early and advanced
ossification) (B) The bone structures distributed in 3 categories (absent, early and advanced os-
sification)
(DOCX)
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
32 / 42


# Página 33

S13 Table. Genes affected in larvae left at 1g relative to those left at 3g for 6 days. The indi-
cates the human homolog of the gene, its "Entrez" gene name, the log ratio of (1g) larvae com-
pared to larvae kept at 3g between 0 and 6dpf, the presence of duplicate probes on the
microarray (D) and the type of protein it encodes. Genes are arranged according to their type
and in alphabetical order.
(DOCX)
S14 Table. Genes affected in larvae returned to 1g on the axis of the centrifuge between day
5–6dpf (3g>axe) relative to those left at 3g for 6 days. The indicates the human homolog of
the gene, its "Entrez" gene name, the log ratio of (3g>axe) larvae compared to larvae kept at 3g
between 0 and 6dpf, the presence of duplicate probes on the microarray (D) and the type of
protein it encodes. Genes are arranged according to their type and in alphabetical order.
(DOCX)
S15 Table. Genes affected in larvae returned to 1g outside of the centrifuge between day
5–6dpf (3g>1g) relative to those left at 3g for 6 days. The indicates the human homolog of
the gene, its "Entrez" gene name, the log ratio of (3g>1g) larvae compared to larvae kept at 3g
between 0 and 6dpf, the presence of duplicate probes on the microarray (D) and the type of
protein it encodes. Genes are arranged according to their type and in alphabetical order.
(DOCX)
S16 Table. Comparison of fold change (FC) values from the microarray dataset with those
observed by RT-qPCR in the "relative microgravity" experiments. The fold change and sta-
tistical significance (p-values) are given from the microarray data and the RT-qPCR confirma-
tion experiments. In the 3g>axe experiment, the human KLF2 gene in S12 is actually the klf2b
zebrafish ortholog, in contrast to the klf2a ortholog shown here.
(DOCX)
S17 Table. Heat map representation of gene regulation in the different conditions. The
gene symbol and name is given, as well as the log(fold-change) values in the different experi-
ments. Induction values are underlined in red (>1) or orange (between 0.378 and 1), repres-
sion values are underlined in blue (-0.378/-1) or green (<-1).
(DOCX)
S18 Table. Heat map representation of canonical pathways affected in the different condi-
tions. The corresponding—Log(p-value) obtained in IPA analysis was used for classification
and are coded by underlining: red means >3, orange between 1 and 3, and yellow means <1.
(DOCX)
S19 Table. Heat map representation of biological functions affected in the different condi-
tions. The corresponding—Log(p-value) obtained in IPA analysis was used for classification
and are coded by underlining: red means >4, orange between 1 and 3, and yellow means <1.
(DOCX)
Acknowledgments
This work was supported by the "Fonds de la Recherche Fondamentale Collective"; 2.4555.99/
2.4542.00/2.4561.10, the SSTC; PAI: P5/35, the University of Liège; GAME project, the Europe-
an Space Agency projects AO-99-LSS-003 and AO-99-LSS-006, the Belgian Space Agency Pro-
dex projects FISH-GSIM and FISH-SIM. M.M. is a "Chercheur Qualifié du F.N.R.S." JvL grant
MG-057 from the Netherlands Organisation for Scientific (NWO) Research Earth and Life Sci-
ences via the Netherlands Space Office NSO). We wish to thank the GIGA-R zebrafish facility
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
33 / 42


# Página 34

for providing zebrafish adults and fertilized eggs and the GIGA-R GenoTranscriptomics plat-
form for DNA sequencing and RNA quality control, and the ESA ESTEC, TEC-MMG LIS Lab,
especially Mr. Alan Dowson for his support.
Author Contributions
Conceived and designed the experiments: JA MM JvL PA LW RNL. Performed the experi-
ments: JA RNL JvL. Analyzed the data: JA RNL ND MM RM NJ. Contributed reagents/materi-
als/analysis tools: JvL RM LW JA. Wrote the paper: JA MM PA JvL RNL.
References
1.
Langheinrich U, Hennen E, Stott G, Vacun G. Zebrafish as a model organism for the identification and
characterization of drugs and genes affecting p53 signaling. Current Biology. 2002; 12(23):2023–8.
PMID: ISI:000179954200020.
2.
Langheinrich U. Zebrafish: a new model on the pharmaceutical catwalk. Bioessays. 2003; 25(9):904–
12. PMID: ISI:000185779600011.
3.
Hill A, Howard CV, Strahle U, Cossins A. Neurodevelopmental defects in zebrafish (Danio rerio) at en-
vironmentally relevant dioxin (TCDD) concentrations. Toxicological Sciences. 2003; 76(2):392–9.
PMID: ISI:000187173500017.
4.
Peterson RT, Link BA, Dowling JE, Schreiber SL. Small molecule developmental screens reveal the
logic and timing of vertebrate development. Proc Natl Acad Sci U S A. 2000; 97(24):12965–9. doi: 10.
1073/pnas.97.24.12965 PMID: 11087852; PubMed Central PMCID: PMC27161.
5.
Yang L, Kemadjou JR, Zinsmeister C, Bauer M, Legradi J, Muller F, et al. Transcriptional profiling re-
veals barcode-like toxicogenomic responses in the zebrafish embryo. Genome biology. 2007; 8(10):
R227. PMID: 17961207.
6.
Schilling TF, Piotrowski T, Grandel H, Brand M, Heisenberg CP, Jiang YJ, et al. Jaw and branchial
arch mutants in zebrafish I: branchial arches. Development. 1996; 123:329–44. Epub 1996/12/01.
PMID: 9007253.
7.
Piotrowski T, Schilling TF, Brand M, Jiang YJ, Heisenberg CP, Beuchle D, et al. Jaw and branchial
arch mutants in zebrafish II: anterior arches and cartilage differentiation. Development. 1996;
123:345–56. PMID: 9007254.
8.
Flores MV, Lam EY, Crosier P, Crosier K. A hierarchy of Runx transcription factors modulate the
onset of chondrogenesis in craniofacial endochondral bones in zebrafish. Dev Dyn. 2006; 235
(11):3166–76. PMID: 17013873.
9.
Komori T. Regulation of bone development and extracellular matrix protein genes by RUNX2. Cell Tis-
sue Res. 2010; 339(1):189–95. Epub 2009/08/04. doi: 10.1007/s00441-009-0832-8 PMID:
19649655.
10.
Verreijdt L, Debiais-Thibaud M, Borday-Birraux V, Van der Heyden C, Sire JY, Huysseune A. Expres-
sion of the dlx gene family during formation of the cranial bones in the zebrafish (Danio rerio): differen-
tial involvement in the visceral skeleton and braincase. Dev Dyn. 2006; 235(5):1371–89. Epub 2006/
03/15. doi: 10.1002/dvdy.20734 PMID: 16534783.
11.
Spoorendonk KM, Peterson-Maduro J, Renn J, Trowe T, Kranenbarg S, Winkler C, et al. Retinoic acid
and Cyp26b1 are critical regulators of osteogenesis in the axial skeleton. Development. 2008; 135
(22):3765–74. Epub 2008/10/18. doi: dev.024034 [pii] doi: 10.1242/dev.024034 PMID: 18927155.
12.
Renn J, Winkler C. Osterix-mCherry transgenic medaka for in vivo imaging of bone formation. Dev
Dyn. 2009; 238(1):241–8. Epub 2008/12/20. doi: 10.1002/dvdy.21836 PMID: 19097055.
13.
Gavaia PJ, Simes DC, Ortiz-Delgado JB, Viegas CS, Pinto JP, Kelsh RN, et al. Osteocalcin and ma-
trix Gla protein in zebrafish (Danio rerio) and Senegal sole (Solea senegalensis): comparative gene
and protein expression during larval development through adulthood. Gene Expr Patterns. 2006; 6
(6):637–52. Epub 2006/02/07. doi: S1567-133X(05)00163-8 [pii] doi: 10.1016/j.modgep.2005.11.010
PMID: 16458082.
14.
Kim YI, Lee S, Jung SH, Kim HT, Choi JH, Lee MS, et al. Establishment of a bone-specific col10a1:
GFP transgenic zebrafish. Molecules and cells. 2013; 36(2):145–50. doi: 10.1007/s10059-013-0117-
7 PMID: 23852131; PubMed Central PMCID: PMC3887955.
15.
Li N, Felber K, Elks P, Croucher P, Roehl HH. Tracking gene expression during zebrafish osteoblast
differentiation. Dev Dyn. 2009; 238(2):459–66. Epub 2009/01/24. doi: 10.1002/dvdy.21838 PMID:
19161246.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
34 / 42


# Página 35

16.
Apschner A, Schulte-Merker S, Witten PE. Not all bones are created equal—using zebrafish and
other teleost species in osteogenesis research. Methods Cell Biol. 2011; 105:239–55. Epub 2011/09/
29. doi: B978-0-12-381320-6.00010–2 [pii] doi: 10.1016/B978-0-12-381320-6.00010–2 PMID:
21951533.
17.
Vanoevelen J, Janssens A, Huitema LF, Hammond CL, Metz JR, Flik G, et al. Trpv5/6 is vital for epi-
thelial calcium uptake and bone formation. FASEB J. 2011; 25(9):3197–207. Epub 2011/06/15. doi:
fj.11-183145 [pii] doi: 10.1096/fj.11-183145 PMID: 21670068.
18.
Huitema LF, Apschner A, Logister I, Spoorendonk KM, Bussmann J, Hammond CL, et al. Entpd5 is
essential for skeletal mineralization and regulates phosphate homeostasis in zebrafish. Proc Natl
Acad Sci U S A. 2012; 109(52):21372–7. Epub 2012/12/14. doi: 1214231110 [pii] doi: 10.1073/pnas.
1214231110 PMID: 23236130; PubMed Central PMCID: PMC3535636.
19.
Apschner A, Huitema LF, Ponsioen B, Peterson-Maduro J, Schulte-Merker S. Zebrafish enpp1 mu-
tants exhibit pathological mineralization, mimicking features of generalized arterial calcification of in-
fancy (GACI) and pseudoxanthoma elasticum (PXE). Disease models & mechanisms. 2014; 7
(7):811–22. doi: 10.1242/dmm.015693 PMID: 24906371.
20.
Brittijn SA, Duivesteijn SJ, Belmamoune M, Bertens LF, Bitter W, de Bruijn JD, et al. Zebrafish devel-
opment and regeneration: new tools for biomedical research. Int J Dev Biol. 2009; 53(5–6):835–50.
Epub 2009/06/27. doi: 082615sb [pii] doi: 10.1387/ijdb.082615sb PMID: 19557689.
21.
Mitchell RE, Huitema LF, Skinner RE, Brunt LH, Severn C, Schulte-Merker S, et al. New tools for
studying osteoarthritis genetics in zebrafish. Osteoarthritis Cartilage. 2013; 21(2):269–78. Epub 2012/
11/20. doi: 10.1016/j.joca.2012.11.004S1063-4584(12)01017-5 [pii]. PMID: 23159952; PubMed Cen-
tral PMCID: PMC3560059.
22.
Mackay EW, Apschner A, Schulte-Merker S. A bone to pick with zebrafish. Bonekey Rep. 2013;
2:445. Epub 2014/01/15. doi: 10.1038/bonekey.2013.179 PMID: 24422140; PubMed Central PMCID:
PMC3844975.
23.
Pietsch J, Bauer J, Egli M, Infanger M, Wise P, Ulbrich C, et al. The effects of weightlessness on the
human organism and mammalian cells. Current molecular medicine. 2011; 11(5):350–64. PMID:
21568935.
24.
Williams D, Kuipers A, Mukai C, Thirsk R. Acclimation during space flight: effects on human physiolo-
gy. CMAJ: Canadian Medical Association journal = journal de l'Association medicale canadienne.
2009; 180(13):1317–23. doi: 10.1503/cmaj.090628 PMID: 19509005; PubMed Central PMCID:
PMC2696527.
25.
Fong K. The next small step. BMJ. 2004; 329(7480):1441–4. doi: 10.1136/bmj.329.7480.1441 PMID:
15604178; PubMed Central PMCID: PMC535972.
26.
Mori S, Mitarai G, Takabayashi A, Usui S, Sakakibara M, Nagatomo M, et al. Evidence of sensory con-
flict and recovery in carp exposed to prolonged weightlessness. Aviation Space and Environmental
Medicine. 1996; 67(3):256–61. PMID: WOS:A1996TY36400010.
27.
deJong HAA, Sondag E, Kuipers A, Oosterveld WJ. Swimming behavior of fish during short periods of
weightlessness. Aviation Space and Environmental Medicine. 1996; 67(5):463–6. PMID: WOS:
A1996UH18500010.
28.
Ohnishi T, Tsuji K, Ohmura T, Matsumoto H, Wang X, Takahahsi A, et al. Accumulation of stress pro-
tein 72 (HSP72) in muscle and spleen of goldfish taken into space. Life Sciences: Microgravity Re-
search. 1998; 21(8–9):1077–80. PMID: WOS:000074330200005.
29.
Takabayashi A, Ohmura-Iwasaki T, Mori S. Changes of vertical eye movements of goldfish for differ-
ent otolith stimulation by linear acceleration. Space Life Sciences: Gravitational Biology: 2002. 2003;
32(8):1527–32. PMID: WOS:000187952900012.
30.
Watanabe S, Takabayashi A, Takagi S, von Baumgarten R, Wetzig J. Dorsal light response and
changes of its responses under varying acceleration conditions. Advances in space research: the offi-
cial journal of the Committee on Space Research. 1989; 9(11):231–40. PMID: INSPEC:3547821.
31.
Suzuki N, Kitamura K, Nemoto T, Shimizu N, Wada S, Kondo T, et al. Effect of vibration on osteoblas-
tic and osteoclastic activities: Analysis of bone metabolism using goldfish scale as a model for bone.
Advances in Space Research. 2007; 40(11):1711–21. PMID: CCC:000253590200022.
32.
Rahmann H, Slenzka K, Hilbig R. Effect of hyper-gravity on the swimming behavior of aquatic verte-
brates. Proceedings of the Fourth European Symposium on Life Sciences Research in Space (ESA
SP-307). 1990:259–63|xiv+663. PMID: INSPEC:3865254.
33.
Slenzka K. Neuroplasticity changes during space flight. Space Life Sciences: Biodosimetry, Biomark-
ers and Late Stochastic Effects of Space Radiation. 2003; 31(6):1595–604. PMID:
WOS:000185090300013.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
35 / 42


# Página 36

34.
Paulus U, Kortje KH, Slenzka K, Rahmann H. Correlation of altered gravity and cytochrome oxidase
activity in the developing fish brain. Journal of Brain Research-Journal Fur Hirnforschung. 1996; 37
(1):103–7. PMID: WOS:A1996UE26200011.
35.
Nindl G, Kortje KH, Slenzka K, Rahmann H. Comparative electronmicroscopical investigations on the
influences of altered gravity on cytochrome oxidase in the inner ear of fish: A spaceflight study. Jour-
nal of Brain Research-Journal Fur Hirnforschung. 1996; 37(3):291–300. PMID: WOS:
A1996VF43300001.
36.
Paulus U, Nindl G, Kortje KH, Slenzka K, Neubert J, Rahmann H. Influence of Altered Gravity on the
Cytochemical-Localization of Cytochrome-Oxidase Activity in Central and Peripheral, Gravisensory
Systems in Developing Cichlid Fish. Life and Gravity: Physiological and Morphological Responses.
1995; 17(6–7):285–8. PMID: WOS:A1995BD67H00043.
37.
Edelmann E, Anken RH, Rahmann H. Swimming behaviour and calcium incorporation into inner ear
otoliths of fish after vestibular nerve transection. Space Life Sciences: Search for Signatures of Life,
and Space Flight Environmental Effects on the Nervous System. 2004; 33(8):1390–4. PMID:
WOS:000222000200030.
38.
Hilbig R, Anken RH, Bauerle A, Rahmann H. Susceptibility to motion sickness in fish: a parabolic air-
craft flight study. Proceedings of `Life in Space for Life on Earth' 8th European Symposium on Life Sci-
ences Research in Space 23rd Annual International Gravitational Physiology Meeting (SP-501).
2002:139–40|xix+437. PMID: INSPEC:7720999.
39.
Beier M. On the influence of altered gravity on the growth of fish inner ear otoliths. Acta astronautica.
1999; 44(7–12):585–91. PMID: CCC:000082555400029.
40.
Piepenbreier K, Renn J, Fischer R, Goerlich R. Influence of space flight conditions on phenotypes and
function of nephritic immune cells of swordtail fish (Xiphophorus helleri). Advances in Space Re-
search. 2006; 38(6):1016–24. PMID: WOS:000202988300002.
41.
Furukawa R, Ijiri K. Swimming behavior of larval medaka fish under microgravity. Space Life Sci-
ences: Biological Research and Space Radiation. 2002; 30(4):733–8. PMID:
WOS:000179730600005.
42.
Ijiri K. Development of space-fertilized eggs and formation of primordial germ cells in the embryos of
Medaka fish. Life Sciences: Microgravity Research. 1998; 21(8–9):1155–8. PMID:
WOS:000074330200017.
43.
Ijiri K, Mizuno R, Eguchi H. Use of an otolith-deficient mutant in studies of fish behavior in microgravity.
Space Life Sciences: Gravitational Biology: 2002. 2003; 32(8):1501–12. PMID:
WOS:000187952900009.
44.
Mizuno R, Ijiri K. Otolith formation in a mutant medaka with a deficiency in gravity-sensing. Space Life
Sciences: Gravitational Biology: 2002. 2003; 32(8):1513–20. PMID: WOS:000187952900010.
45.
Shimomura-Umemura S, Ijiri K. Effect of hypergravity on expression of the immediate early gene, c-
fos, in central nervous system of medaka (Oryzias latipes). Advances in Space Research. 2006; 38
(6):1082–8. PMID: WOS:000202988300011.
46.
Niihori M, Mogami Y, Naruse K, Baba SA. Development and swimming behavior of medaka fry in a
spaceflight aboard the space shuttle Columbia (STS-107). Zoological science. 2004; 21(9):923–31.
PMID: WOS:000237374500003.
47.
Moorman SJ, Burress C, Cordova R, Slater J. Stimulus dependence of the development of the zebra-
fish (Danio rerio) vestibular system. Journal of neurobiology. 1999; 38(2):247–58. PMID:
WOS:000078159900007.
48.
Wiederhold ML, Harrison JL, Gao WY. A critical period for gravitational effects on otolith formation.
Journal of Vestibular Research-Equilibrium & Orientation. 2003; 13(4–6):205–14. PMID:
WOS:000221711500005.
49.
Shimada N, Moorman SJ. Changes in gravitational force cause changes in gene expression in the
lens of developing zebrafish. Developmental Dynamics. 2006; 235(10):2686–94. PMID:
CCC:000240906200006.
50.
Shimada N, Sokunbi G, Moorman SJ. Changes in gravitational force affect gene expression in devel-
oping organ systems at different developmental times—art. no. 10. Bmc Developmental Biology.
2005; 5:10–. PMID: CCC:000234277900001.
51.
Edsall SC, Franz-Odendaal TA. An assessment of the long-term effects of simulated microgravity on
cranial neural crest cells in zebrafish embryos with a focus on the adult skeleton. PLoS One. 2014; 9
(2):e89296. doi: 10.1371/journal.pone.0089296 PMID: 24586670; PubMed Central PMCID:
PMC3930699.
52.
Kondrachuk AV, Boyle RD. Feedback hypothesis and the effects of altered gravity on formation and
function of gravireceptors of mollusks and fish. Archives Italiennes De Biologie. 2006; 144(2):75–87.
PMID: WOS:000236817500002.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
36 / 42


# Página 37

53.
Rahmann H, Anken RH. Gravitational neurobiology of fish. Life Sciences: Microgravity and Space Ra-
diation Effects. 2000; 25(10):1985–95. PMID: WOS:000085822600001.
54.
Asai Y, Starr CJ, Kappler JA, Chan DK, Pataky F, Kollmar R, et al. A zebrafish mutation affecting fibro-
blast growth factor signaling in the inner ear. Faseb Journal. 2005; 19(5):A1367–A8. PMID:
WOS:000227610902509.
55.
Lathers CM, Mukai C, Smith CM, Schraeder PL. A new goldfish model to evaluate pharmacokinetic
and pharmacodynamic effects of drugs used for motion sickness in different gravity loads. Acta astro-
nautica. 2001; 49(3–10):419–40. PMID: CCC:000170751900031.
56.
Anken RH, Hilbig R. A drop-tower experiment to determine the threshold of gravity for inducing motion
sickness in fish. Space Life Sciences: Life Support Systems and Biological Systems under Influence
of Physical Factors. 2004; 34(7):1592–7. PMID: WOS:000225592800018.
57.
Anken R, Forster A, Baur U, Feucht I, Hilbig R. Otolith asymmetry and kinetotic behaviour of fish at
high-quality microgravity: A drop-tower experiment. Advances in Space Research. 2006; 38(6):1032–
6. PMID: WOS:000202988300004.
58.
Marthy HJ. Developmental biology of animal models under varied gravity conditions: A review. Vie Et
Milieu-Life and Environment. 2002; 52(4):149–66. PMID: CCC:000180697800005.
59.
Van Loon J, Krause J, Cunha H, Goncalves J, Almeida H, Schiller P. The Large Diameter Centrifuge,
LDC, for life and physical sciences and technology. Proc of the 'Life in Space for Life on Earth Sympo-
sium', Angers, France, 22–27 June 2008 ESA SP-663, December 2008. 2008.
60.
Van Loon J. The Application of Centrifuges in "Reduced Gravity" Research. Session F12-0018-10
38th Assembly of the Committee on Space Research (COSPAR), Bremen, Germany, July 2010.
2010.
61.
Westerfield M. THE ZEBRAFISH BOOK, 5th Edition; A guide for the laboratory use of zebrafish
(Danio rerio), Eugene, University of Oregon Press. 2007.
62.
Kimmel CB, Ballard WW, Kimmel SR, Ullmann B, Schilling TF. Stages of embryonic development of
the zebrafish. Dev Dyn. 1995; 203(3):253–310. PMID: 8589427.
63.
Fleming A, Sato M, Goldsmith P. High-throughput in vivo screening for bone anabolic compounds
with zebrafish. Journal of biomolecular screening. 2005; 10(8):823–31. PMID: ISI:000234407000007.
64.
Van Loon J. The Gravity Environment in Space Experiments. in “Biology in Space and Life on Earth”
Editor Brinckmann E Whiley, Weinheim, Germany. 2007:17–32.
65.
Walker MB, Kimmel CB. A two-color acid-free cartilage and bone stain for zebrafish larvae. Biotech
Histochem. 2007; 82(1):23–8. Epub 2007/05/19. doi: 778819433 [pii] doi: 10.1080/
10520290701333558 PMID: 17510811.
66.
Kimmel CB, Miller CT, Kruze G, Ullmann B, BreMiller RA, Larison KD, et al. The shaping of pharyn-
geal cartilages during early development of the zebrafish. Dev Biol. 1998; 203(2):245–63. PMID:
9808777.
67.
Cubbage CC, Mabee PM. Development of the Cranium and Paired Fins in the Zebrafish Danio rerio
(Ostariophysi, Cyprinidae). J Morphol. 1996; 229:121–60.
68.
Schilling TF. The morphology of larval and adult zebrafish. Zebrafish, a practical approach Eds: Nüss-
lein-Volhard C; Dahm R Oxford University Press. 2002:59–94.
69.
Marée R, Stevens B, Rollus L, Rocks N, Moles-Lopez X, Salmon I, et al. A rich internet application for
remote visualization and collaborative annotation of digital slide images in histology and cytology. Di-
agnostic Pathology. 8(S1)2013. p. S26–S9.
70.
Pfaffl MW. A new mathematical model for relative quantification in real-time RT-PCR. Nucleic Acids
Res. 2001; 29(9):2002–7. Epub 2001/05/09. PMID: 11328886; PubMed Central PMCID: PMC55695.
71.
Nourizadeh-Lillabadi R, Lyche JL, Almaas C, Stavik B, Moe SJ, Aleksandersen M, et al. Transcription-
al regulation in liver and testis associated with developmental and reproductive effects in male zebra-
fish exposed to natural mixtures of persistent organic pollutants (POP). Journal of toxicology and
environmental health Part A. 2009; 72(3–4):112–30. doi: 10.1080/15287390802537255 PMID:
19184727.
72.
Smyth GK, Speed T. Normalization of cDNA microarray data. Methods. 2003; 31(4):265–73. PMID:
14597310.
73.
Smyth GK, Michaud J, Scott HS. Use of within-array replicate spots for assessing differential expres-
sion in microarray experiments. Bioinformatics. 2005; 21(9):2067–75. doi: 10.1093/bioinformatics/
bti270 PMID: 15657102.
74.
Benjamini Y, Hochberg MC. Controlling the false discovery rate: A practical and powerful approach to
multiple testing. J R Stat Soc Ser B (Methodological). 1995; 57:289–300.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
37 / 42


# Página 38

75.
Shih TH, Horng JL, Liu ST, Hwang PP, Lin LY. Rhcg1 and NHE3b are involved in ammonium-depen-
dent sodium uptake by zebrafish larvae acclimated to low-sodium water. American journal of physiolo-
gy Regulatory, integrative and comparative physiology. 2012; 302(1):R84–93. doi: 10.1152/ajpregu.
00318.2011 PMID: 21993530.
76.
Chang WJ, Wang YF, Hu HJ, Wang JH, Lee TH, Hwang PP. Compensatory regulation of Na+ absorp-
tion by Na+/H+ exchanger and Na+-Cl- cotransporter in zebrafish (Danio rerio). Frontiers in zoology.
2013; 10(1):46. doi: 10.1186/1742-9994-10-46 PMID: 23924428; PubMed Central PMCID:
PMC3750650.
77.
Hwang PP, Chou MY. Zebrafish as an animal model to study ion homeostasis. Pflugers Archiv: Euro-
pean journal of physiology. 2013; 465(9):1233–47. doi: 10.1007/s00424-013-1269-1 PMID:
23568368; PubMed Central PMCID: PMC3745619.
78.
Jongen JW, Willemstein-van Hove EC, van der Meer JM, Bos MP, Juppner H, Segre GV, et al. Down-
regulation of the receptor for parathyroid hormone (PTH) and PTH-related peptide by PTH in primary
fetal rat osteoblasts. J Bone Miner Res. 1996; 11(9):1218–25. doi: 10.1002/jbmr.5650110905 PMID:
8864895.
79.
Kawane T, Mimura J, Yanagawa T, Fujii-Kuriyama Y, Horiuchi N. Parathyroid hormone (PTH) down-
regulates PTH/PTH-related protein receptor gene expression in UMR-106 osteoblast-like cells via a
3',5'-cyclic adenosine monophosphate-dependent, protein kinase A-independent pathway. J Endocri-
nol. 2003; 178(2):247–56. PMID: 12904172.
80.
Farioli-Vecchioli S, Saraulli D, Costanzi M, Leonardi L, Cina I, Micheli L, et al. Impaired terminal differ-
entiation of hippocampal granule neurons and defective contextual memory in PC3/Tis21 knockout
mice. PLoS One. 2009; 4(12):e8339. doi: 10.1371/journal.pone.0008339 PMID: 20020054; PubMed
Central PMCID: PMC2791842.
81.
Dattani MT. Growth hormone deficiency and combined pituitary hormone deficiency: does the geno-
type matter? Clin Endocrinol (Oxf). 2005; 63(2):121–30. Epub 2005/08/03. doi: CEN2289 [pii] doi: 10.
1111/j.1365-2265.2005.02289.x PMID: 16060904.
82.
Salie R, Kneissel M, Vukevic M, Zamurovic N, Kramer I, Evans G, et al. Ubiquitous overexpression of
Hey1 transcription factor leads to osteopenia and chondrocyte hypertrophy in bone. Bone. 2010; 46
(3):680–94. doi: 10.1016/j.bone.2009.10.022 PMID: 19857617.
83.
Horn E, van Loon JJWA, Aceto J, Muller M. Life Sciences: Animal Physiology. Laboratory Science
with Space Data. 2011;Eds. Beysens D., Carotenuto L., van Loon J. J. W.A., Zell M.; ISBN 978-3-
642-21143-0 Springer Verlag Berlin Heidelberg 2011:123–9.
84.
Muller M, Dalcq J, Aceto J, Larbuisson A, Pasque V, Nourizadeh-Lillidadi R, et al. The function of the
Egr1 transcrition factor in cartilage formation and adaptation to microgravity in Danio rerio. J Appl
Ichthyol. 2010; 26:239–44.
85.
Muller M, Dalcq J, Pasque V, Aceto J, Motte P, Martial JA. The function of the Egr1 transcription factor
in cartilage formation and adaptation to microgravity in Danio rerio. Journal of gravitational physiology:
a journal of the International Society for Gravitational Physiology. 2009; 16:in press.
86.
Aceto J, Nourizadeh-Lilladadi R, van Loon J, Motte P, Alestrom P, Martial JA, et al. Microgravity simu-
lation comparison at genome level in Danio rerio and role of Sox4 transcription factors in cranial skele-
ton development. Journal of gravitational physiology: a journal of the International Society for
Gravitational Physiology. 2009; 16:in press.
87.
Muller M, Aceto J, Dalcq J, Alestrom P, Nourizadeh-Lillabadi R, Goerlich R, et al. Small Fish Species
as Powerful Model Systems to Study Vertebrate Physiology in Space. J Gravit Physiol. 2008; 15:253–
4.
88.
Aceto J, Muller M, Nourizadeh-Lillabadi R, Alestrom P, Van Loon J, Schiller V, et al. Small fish species
as powerful model systems to study vertebrate physiology in space. Journal of gravitational physiolo-
gy: a journal of the International Society for Gravitational Physiology. 2008; 15:111–2.
89.
Goerlich R, Renn J, Alestrom P, Nourizadeh-Lillabadi R, Schartl M, Winkler C, et al. European Net-
work Using Fish as Osteoporosis Research Model (ENFORM). J Grav Physiol. 2005; 12(1):P279–
P80.
90.
Lin CH, Su CH, Tseng DY, Ding FC, Hwang PP. Action of vitamin D and the receptor, VDRa, in calci-
um handling in zebrafish (Danio rerio). PLoS One. 2012; 7(9):e45650. doi: 10.1371/journal.pone.
0045650 PMID: 23029160; PubMed Central PMCID: PMC3446910.
91.
Witten PE, Huysseune A. A comparative view on mechanisms and functions of skeletal remodelling in
teleost fish, with special emphasis on osteoclasts and their function. Biological reviews of the Cam-
bridge Philosophical Society. 2009; 84(2):315–46. Epub 2009/04/23. doi: BRV77 [pii] doi: 10.1111/j.
1469-185X.2009.00077.x PMID: 19382934.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
38 / 42


# Página 39

92.
Okabe M, Graham A. The origin of the parathyroid gland. Proc Natl Acad Sci U S A. 2004; 101
(51):17716–9. doi: 10.1073/pnas.0406116101 PMID: 15591343; PubMed Central PMCID:
PMC539734.
93.
Zajac JD, Danks JA. The development of the parathyroid gland: from fish to human. Current opinion in
nephrology and hypertension. 2008; 17(4):353–6. doi: 10.1097/MNH.0b013e328304651c PMID:
18660669.
94.
Suzuki N, Danks JA, Maruyama Y, Ikegame M, Sasayama Y, Hattori A, et al. Parathyroid hormone 1
(1–34) acts on the scales and involves calcium metabolism in goldfish. Bone. 2011; 48(5):1186–93.
doi: 10.1016/j.bone.2011.02.004 PMID: 21334472.
95.
Danks JA, D'Souza DG, Gunn HJ, Milley KM, Richardson SJ. Evolution of the parathyroid hormone
family and skeletal formation pathways. Gen Comp Endocrinol. 2011; 170(1):79–91. doi: 10.1016/j.
ygcen.2010.10.023 PMID: 21074535.
96.
Fuentes J, Haond C, Guerreiro PM, Silva N, Power DM, Canario AV. Regulation of calcium balance in
the sturgeon Acipenser naccarii: a role for PTHrP. American journal of physiology Regulatory, integra-
tive and comparative physiology. 2007; 293(2):R884–93. doi: 10.1152/ajpregu.00203.2007 PMID:
17491110.
97.
Yan YL, Bhattacharya P, He XJ, Ponugoti B, Marquardt B, Layman J, et al. Duplicated zebrafish co-
orthologs of parathyroid hormone-related peptide (PTHrP, Pthlh) play different roles in craniofacial
skeletogenesis. J Endocrinol. 2012; 214(3):421–35. doi: 10.1530/JOE-12-0110 PMID: 22761277;
PubMed Central PMCID: PMC3718479.
98.
Fuentes J, Guerreiro PM, Modesto T, Rotllant J, Canario AV, Power DM. A PTH/PTHrP receptor an-
tagonist blocks the hypercalcemic response to estradiol-17beta. American journal of physiology Reg-
ulatory, integrative and comparative physiology. 2007; 293(2):R956–60. doi: 10.1152/ajpregu.00111.
2007 PMID: 17537843.
99.
Schein V, Cardoso JC, Pinto PI, Anjos L, Silva N, Power DM, et al. Four stanniocalcin genes in teleost
fish: structure, phylogenetic analysis, tissue distribution and expression during hypercalcemic chal-
lenge. Gen Comp Endocrinol. 2012; 175(2):344–56. doi: 10.1016/j.ygcen.2011.11.033 PMID:
22154646.
100.
Fuentes J, Power DM, Canario AV. Parathyroid hormone-related protein-stanniocalcin antagonism in
regulation of bicarbonate secretion and calcium precipitation in a marine fish intestine. American jour-
nal of physiology Regulatory, integrative and comparative physiology. 2010; 299(1):R150–8. doi: 10.
1152/ajpregu.00378.2009 PMID: 20410471.
101.
Fleming A, Sato M. Continuous and intermittent treatment of zebrafish with recombinant human para-
thyroid hormone (1–34) have opposite effects on the growing fish skeleton. Journal of Bone and Min-
eral Research. 2004; 19:S334–S5. PMID: ISI:000224326802034.
102.
Anjos L, Gomes AS, Redruello B, Reinhardt R, Canario AV, Power DM. PTHrP-induced modifications
of the sea bream (Sparus auratus) vertebral bone proteome. Gen Comp Endocrinol. 2013; 191:102–
12. doi: 10.1016/j.ygcen.2013.05.014 PMID: 23747812.
103.
Sebastian C, Esseling K, Horn E. Altered gravitational forces affect the development of the static vesti-
buloocular reflex in fish (Oreochromis mossambicus). Journal of neurobiology. 2001; 46(1):59–72.
PMID: CCC:000166094800006.
104.
Beier M, Anken RH, Rahmann H. Influence of hypergravity on fish inner ear otoliths: II. Incorporation
of calcium and kinetotic behaviour. Space Life Sciences: Biological Research and Space Radiation.
2002; 30(4):727–31. PMID: WOS:000179730600004.
105.
Horn ER. The development of gravity sensory systems during periods of altered gravity dependent
sensory input. Advances in space biology and medicine. 2003; 9:133–71. Epub 2003/11/25. PMID:
14631632.
106.
Anken RH, Beier M, Rahmann H. Hypergravity decreases carbonic anhydrase-reactivity in inner ear
maculae of fish. Journal of experimental zoology Part A, Comparative experimental biology. 2004;
301(10):815–9. Epub 2004/09/28. doi: 10.1002/jez.a.97 PMID: 15449341.
107.
Anken RH. On the role of the central nervous system in regulating the mineralisation of inner-ear oto-
liths of fish. Protoplasma. 2006; 229(2–4):205–8. Epub 2006/12/21. doi: 10.1007/s00709-006-0219-6
PMID: 17180502.
108.
Arias JL, Nakamura O, Fernandez MS, Wu JJ, Knigge P, Eyre DR, et al. Role of type X collagen on
experimental mineralization of eggshell membranes. Connective tissue research. 1997; 36(1):21–33.
PMID: 9298621.
109.
Seitz S, Rendenbach C, Barvencik F, Streichert T, Jeschke A, Schulze J, et al. Retinol deprivation
partially rescues the skeletal mineralization defects of Phex-deficient Hyp mice. Bone. 2013; 53
(1):231–8. doi: 10.1016/j.bone.2012.12.009 PMID: 23266491.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
39 / 42


# Página 40

110.
Komori T, Yagi H, Nomura S, Yamaguchi A, Sasaki K, Deguchi K, et al. Targeted disruption of Cbfa1
results in a complete lack of bone formation owing to maturational arrest of osteoblasts. Cell. 1997; 89
(5):755–64. PMID: 9182763.
111.
Hogan BM, Danks JA, Layton JE, Hall NE, Heath JK, Lieschke GJ. Duplicate zebrafish pth genes are
expressed along the lateral line and in the central nervous system during embryogenesis. Endocrinol-
ogy. 2005; 146(2):547–51. PMID: 15539562.
112.
Craig TA, Zhang Y, McNulty MS, Middha S, Ketha H, Singh RJ, et al. Research resource: whole tran-
scriptome RNA sequencing detects multiple 1alpha,25-dihydroxyvitamin D(3)-sensitive metabolic
pathways in developing zebrafish. Mol Endocrinol. 2012; 26(9):1630–42. doi: 10.1210/me.2012-1113
PMID: 22734042; PubMed Central PMCID: PMC3434529.
113.
Van Loon J, Tanck E, van Nieuwenhoven FA, Snoeckx LHEH, de Jong HAA, Wubbels RJ. A brief
overview of animal hypergravity studies. Journal of gravitational physiology: a journal of the Interna-
tional Society for Gravitational Physiology. 2005; 12:P5–P10.
114.
Van Loon J. Hypergravity studies in the Netherlands. Journal of gravitational physiology: a journal of
the International Society for Gravitational Physiology. 2001; 8:P139–P42. PMID: 12650205
115.
Moorman SJ, Shimada N, Sokunbi G, Pfirrmann C. Simulated-microgravity induced changes in gene
expression in zebrafish embryos suggest that the primary cilium is involved in gravity transduction.
Gravitational Space Biol. 2007; 20(2):79–86.
116.
van Straaten F, Muller R, Curran T, Van Beveren C, Verma IM. Complete nucleotide sequence of a
human c-onc gene: deduced amino acid sequence of the human c-fos protein. Proc Natl Acad Sci U S
A. 1983; 80(11):3183–7. PMID: 6574479; PubMed Central PMCID: PMC394004.
117.
Wang ZQ, Ovitt C, Grigoriadis AE, Mohle-Steinlein U, Ruther U, Wagner EF. Bone and haematopoie-
tic defects in mice lacking c-fos. Nature. 1992; 360(6406):741–5. doi: 10.1038/360741a0 PMID:
1465144.
118.
de Groot RP, Rijken PJ, den Hertog J, Boonstra J, Verkleij AJ, de Laat SW, et al. Microgravity de-
creases c-fos induction and serum response element activity. J Cell Sci. 1990; 97 (Pt 1):33–8. PMID:
2258390.
119.
de Groot RP, Rijken PJ, den Hertog J, Boonstra J, Verkleij AJ, de Laat SW, et al. Nuclear responses
to protein kinase C signal transduction are sensitive to gravity changes. Exp Cell Res. 1991; 197
(1):87–90. PMID: 1915667.
120.
Hughes-Fulford M, Tjandrawinata R, Fitzgerald J, Gasuad K, Gilbertson V. Effects of microgravity on
osteoblast growth. Gravitational and space biology bulletin: publication of the American Society for
Gravitational and Space Biology. 1998; 11(2):51–60. PMID: 11540639.
121.
Sato A, Hamazaki T, Oomura T, Osada H, Kakeya M, Watanabe M, et al. Effects of microgravity on c-
fos gene expression in osteoblast-like MC3T3-E1 cells. Advances in space research: the official jour-
nal of the Committee on Space Research. 1999; 24(6):807–13. PMID: 11542626.
122.
Nose K, Shibanuma M. Induction of early response genes by hypergravity in cultured mouse osteo-
blastic cells (MC3T3-E1). Exp Cell Res. 1994; 211(1):168–70. doi: 10.1006/excr.1994.1073 PMID:
7510248.
123.
Fitzgerald J, Hughes-Fulford M. Gravitational loading of a simulated launch alters mRNA expression
in osteoblasts. Exp Cell Res. 1996; 228(1):168–71. doi: 10.1006/excr.1996.0313 PMID: 8892985.
124.
Fuller CA, Murakami DM, Hoban-Higgins TM, Tang IH. Changes in hypothalamic [correction of
hypothalmic] staining for c-Fos following 2G exposure in rats. Journal of gravitational physiology: a
journal of the International Society for Gravitational Physiology. 1994; 1(1):P69–70. PMID: 11538768.
125.
Gustave Dit Duflo S, Gestreau C, Lacour M. Fos expression in the rat brain after exposure to gravito-
inertial force changes. Brain Res. 2000; 861(2):333–44. PMID: 10760495.
126.
Pompeiano O, d'Ascanio P, Centini C, Pompeiano M, Balaban E. Gene expression in rat vestibular
and reticular structures during and after space flight. Neuroscience. 2002; 114(1):135–55. PMID:
12207961.
127.
Nakagawa A, Uno A, Horii A, Kitahara T, Kawamoto M, Uno Y, et al. Fos induction in the amygdala by
vestibular information during hypergravity stimulation. Brain Res. 2003; 986(1–2):114–23. PMID:
12965235.
128.
Kaufman GD. Fos expression in the vestibular brainstem: what one marker can tell us about the net-
work. Brain research Brain research reviews. 2005; 50(1):200–11. doi: 10.1016/j.brainresrev.2005.
06.001 PMID: 16039721.
129.
Brown JR, Ye H, Bronson RT, Dikkes P, Greenberg ME. A defect in nurturing in mice lacking the im-
mediate early gene fosB. Cell. 1996; 86(2):297–309. PMID: 8706134.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
40 / 42


# Página 41

130.
Lee SL, Sadovsky Y, Swirnoff AH, Polish JA, Goda P, Gavrilina G, et al. Luteinizing hormone deficien-
cy and female infertility in mice lacking the transcription factor NGFI-A (Egr-1). Science. 1996; 273
(5279):1219–21. PMID: 8703054.
131.
Topilko P, Schneider-Maunoury S, Levi G, Trembleau A, Gourdji D, Driancourt MA, et al. Multiple pitu-
itary and ovarian defects in Krox-24 (NGFI-A, Egr-1)-targeted mice. Mol Endocrinol. 1998; 12(1):107–
22. PMID: 9440815.
132.
Granet C, Boutahar N, Vico L, Alexandre C, Lafage-Proust MH. MAPK and SRC-kinases control
EGR-1 and NF-kappa B inductions by changes in mechanical environment in osteoblasts. Biochem
Biophys Res Commun. 2001; 284(3):622–31. Epub 2001/06/09. doi: 10.1006/bbrc.2001.5023S0006-
291X(01)95023-5 [pii]. PMID: 11396946.
133.
Close R, Toro S, Martial JA, Muller M. Expression of the zinc finger Egr1 gene during zebrafish embry-
onic development. Mech Dev. 2002; 118(1–2):269–72. PMID: 12351200.
134.
Dalcq J, Pasque V, Ghaye A, Larbuisson A, Motte P, Martial JA, et al. Runx3, Egr1 and Sox9b form a
regulatory cascade required to modulate BMP-signaling during cranial cartilage development in zeb-
rafish. PLoS One. 2012; 7(11):e50140. Epub 2012/12/05. doi: 10.1371/journal.pone.0050140PONE-
D-12-13904 [pii]. PMID: 23209659; PubMed Central PMCID: PMC3507947.
135.
Larbuisson A, Dalcq J, Martial JA, Muller M. Fgf receptors Fgfr1a and Fgfr2 control the function of pha-
ryngeal endoderm in late cranial cartilage development. Differentiation. 2013; 86:192–206. Epub
2013 Oct 29. doi: 10.1016 PMID: 24176552
136.
Kurihara Y, Kurihara H, Suzuki H, Kodama T, Maemura K, Nagai R, et al. Elevated blood pressure
and craniofacial abnormalities in mice deficient in endothelin-1. Nature. 1994; 368(6473):703–10. doi:
10.1038/368703a0 PMID: 8152482.
137.
Miller CT, Schilling TF, Lee K, Parker J, Kimmel CB. sucker encodes a zebrafish Endothelin-1 re-
quired for ventral pharyngeal arch development. Development. 2000; 127(17):3815–28. Epub 2000/
08/10. PMID: 10934026.
138.
Roberts AW, Robb L, Rakar S, Hartley L, Cluse L, Nicola NA, et al. Placental defects and embryonic
lethality in mice lacking suppressor of cytokine signaling 3. Proc Natl Acad Sci U S A. 2001; 98
(16):9324–9. doi: 10.1073/pnas.161271798 PMID: 11481489; PubMed Central PMCID: PMC55419.
139.
Liang J, Wang D, Renaud G, Wolfsberg TG, Wilson AF, Burgess SM. The stat3/socs3a pathway is a
key regulator of hair cell regeneration in zebrafish. [corrected]. J Neurosci. 2012; 32(31):10662–73.
doi: 10.1523/JNEUROSCI.5785-10.2012 PMID: 22855815; PubMed Central PMCID: PMC3427933.
140.
Lu B, Ferrandino AF, Flavell RA. Gadd45beta is important for perpetuating cognate and inflammatory
signals in T cells. Nature immunology. 2004; 5(1):38–44. doi: 10.1038/ni1020 PMID: 14691480.
141.
Wani MA, Means RT Jr., Lingrel JB. Loss of LKLF function results in embryonic lethality in mice.
Transgenic research. 1998; 7(4):229–38. PMID: 9859212.
142.
Vermot J, Forouhar AS, Liebling M, Wu D, Plummer D, Gharib M, et al. Reversing blood flows act
through klf2a to ensure normal valvulogenesis in the developing heart. PLoS Biol. 2009; 7(11):
e1000246. doi: 10.1371/journal.pbio.1000246 PMID: 19924233; PubMed Central PMCID:
PMC2773122.
143.
Wang L, Zhang P, Wei Y, Gao Y, Patient R, Liu F. A blood flow-dependent klf2a-NO signaling cascade
is required for stabilization of hematopoietic stem cell programming in zebrafish embryos. Blood.
2011; 118(15):4102–10. doi: 10.1182/blood-2011-05-353235 PMID: 21849483.
144.
Renn J, Pruvot B, Muller M. Detection of nitric oxide by diaminofluorescein visualizes the skeleton in
living zebrafish. J Appl Ichthyol. 2014; 30:701–6.
145.
Henrotin YE, Bruckner P, Pujol JP. The role of reactive oxygen species in homeostasis and degrada-
tion of cartilage. Osteoarthr Cartilage. 2003; 11(10):747–55. Epub 2003/09/18. doi:
S106345840300150X [pii]. PMID: 13129694.
146.
Saura M, Tarin C, Zaragoza C. Recent insights into the implication of nitric oxide in osteoblast differen-
tiation and proliferation during bone development. TheScientificWorldJ. 2010; 10:624–32.
147.
Britsch S, Goerich DE, Riethmacher D, Peirano RI, Rossner M, Nave KA, et al. The transcription fac-
tor Sox10 is a key regulator of peripheral glial development. Genes Dev. 2001; 15(1):66–78. PMID:
11156606; PubMed Central PMCID: PMC312607.
148.
Dutton K, Abbas L, Spencer J, Brannon C, Mowbray C, Nikaido M, et al. A zebrafish model for Waar-
denburg syndrome type IV reveals diverse roles for Sox10 in the otic vesicle. Disease models &
mechanisms. 2009; 2(1–2):68–83. doi: 10.1242/dmm.001164 PMID: 19132125; PubMed Central
PMCID: PMC2615172.
149.
Whitfield TT, Granato M, van Eeden FJ, Schach U, Brand M, Furutani-Seiki M, et al. Mutations affect-
ing development of the zebrafish inner ear and lateral line. Development. 1996; 123:241–54. PMID:
9007244.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
41 / 42


# Página 42

150.
Malicki J, Schier AF, Solnica-Krezel L, Stemple DL, Neuhauss SC, Stainier DY, et al. Mutations affect-
ing development of the zebrafish ear. Development. 1996; 123:275–83. PMID: 9007247.
151.
Cau E, Gradwohl G, Casarosa S, Kageyama R, Guillemot F. Hes genes regulate sequential stages of
neurogenesis in the olfactory epithelium. Development. 2000; 127(11):2323–32. PMID: 10804175.
152.
Karlsson C, Jonsson M, Asp J, Brantsing C, Kageyama R, Lindahl A. Notch and HES5 are regulated
during human cartilage differentiation. Cell Tissue Res. 2007; 327(3):539–51. doi: 10.1007/s00441-
006-0307-0 PMID: 17093926.
153.
Trahey M, Weissman IL. Cyclophilin C-associated protein: a normal secreted glycoprotein that down-
modulates endotoxin and proinflammatory responses in vivo. Proc Natl Acad Sci U S A. 1999; 96
(6):3006–11. PMID: 10077627; PubMed Central PMCID: PMC15885.
154.
Zheng CL, Sumizawa T, Che XF, Tsuyama S, Furukawa T, Haraguchi M, et al. Characterization of
MVP and VPARP assembly into vault ribonucleoprotein complexes. Biochem Biophys Res Commun.
2005; 326(1):100–7. doi: 10.1016/j.bbrc.2004.11.006 PMID: 15567158.
155.
Blaker-Lee A, Gupta S, McCammon JM, De Rienzo G, Sive H. Zebrafish homologs of genes within
16p11.2, a genomic region associated with brain disorders, are active during brain development, and
include two deletion dosage sensor genes. Disease models & mechanisms. 2012; 5(6):834–51. doi:
10.1242/dmm.009944 PMID: 22566537; PubMed Central PMCID: PMC3484866.
156.
Hammond TG, Benes E, O'Reilly KC, Wolf DA, Linnehan RM, Taher A, et al. Mechanical culture con-
ditions effect gene expression: gravity-induced changes on the space shuttle. Physiological geno-
mics. 2000; 3(3):163–73. PMID: 11015612.
Zebrafish Bone and General Physiology in Hyper-Gravity
PLOS ONE | DOI:10.1371/journal.pone.0126928
June 10, 2015
42 / 42
