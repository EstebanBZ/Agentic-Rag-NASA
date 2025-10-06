# Página 1

RESEARCH ARTICLE
A microRNA signature and TGF-β1 response
were identified as the key master regulators
for spaceflight response
Afshin Beheshti1☯*, Shayoni Ray2☯, Homer Fogle1, Daniel Berrios2, Sylvain V. Costes3*
1 WYLE, NASA Ames Research Center, Moffett Field, California, United States of America, 2 USRA, NASA
Ames Research Center, Moffett Field, California, United States of America, 3 NASA Ames Research Center,
Space Biosciences Division, Moffett Field, California, United States of America
☯These authors contributed equally to this work.
* afshin.beheshti@nasa.gov (AB); sylvain.v.costes@nasa.gov (SVC)
Abstract
Translating fundamental biological discoveries from NASA Space Biology program into
health risk from space flights has been an ongoing challenge. We propose to use NASA
GeneLab database to gain new knowledge on potential systemic responses to space. Unbi-
ased systems biology analysis of transcriptomic data from seven different rodent datasets
reveals for the first time the existence of potential “master regulators” coordinating a sys-
temic response to microgravity and/or space radiation with TGF-β1 being the most common
regulator. We hypothesized the space environment leads to the release of biomolecules cir-
culating inside the blood stream. Through datamining we identified 13 candidate microRNAs
(miRNA) which are common in all studies and directly interact with TGF-β1 that can be
potential circulating factors impacting space biology. This study exemplifies the utility of the
GeneLab data repository to aid in the process of performing novel hypothesis–based
research.
Introduction
The spaceflight environment has been evaluated for human health risks due to hazardous fac-
tors such as ionizing radiation, microgravity, hypoxia, hypothermia as well as other associated
physiological/psychological stressors. Space radiation and microgravity have been considered
as the primary potential show-stoppers for long duration space exploration missions beyond
LEO (Low Earth Orbit) and are thought to be driving many of the responses observed in flown
organisms. Due to the high cost and limited availability of larger cohort for spaceflight sam-
ples, rodents have emerged as the primary choice among the model organisms for flight exper-
iments. This is due to its similarity to the human genome [1] and their relative small size
compared to other rodents. Omics data related to these studies have been made available to
the public through NASA’s GeneLab platform (genelab.nasa.gov). GeneLab is an open
repository that houses fully coordinated and curated experimental results, raw data and
metadata pertaining to model organisms onboard International Space Station (ISS), Space
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
1 / 19
a1111111111
a1111111111
a1111111111
a1111111111
a1111111111
OPEN ACCESS
Citation: Beheshti A, Ray S, Fogle H, Berrios D,
Costes SV (2018) A microRNA signature and TGF-
β1 response were identified as the key master
regulators for spaceflight response. PLoS ONE 13
(7): e0199621. https://doi.org/10.1371/journal.
pone.0199621
Editor: Andre van Wijnen, University of
Massachusetts Medical School, UNITED STATES
Received: March 6, 2018
Accepted: May 3, 2018
Published: July 25, 2018
Copyright: This is an open access article, free of all
copyright, and may be freely reproduced,
distributed, transmitted, modified, built upon, or
otherwise used by anyone for any lawful purpose.
The work is made available under the Creative
Commons CC0 public domain dedication.
Data Availability Statement: All raw data used in
this manuscript is available on https://genelab.
nasa.gov/.
Funding: Research funding was provided by the
GeneLab Project at NASA Ames Research Center,
through NASA’s Space Biology Program in the
Division of Space Life and Physical Sciences
Research and Applications (SLPSRA). Any use of
trade names is for descriptive purposes only and
does not imply endorsement by the US
Government.


# Página 2

Transportation Systems (STS), or the russian animal carry capsule called Bion-M1 [2]. Gene-
Lab also houses ground NASA studies which model spaceflight environment. The vast
amounts of genomic, proteomic, transcriptomic and metabolomic data from several organ-
isms and cell culture experiments are available to researchers across the globe for in silico anal-
ysis and generating hypothesis-driven future research.
Several mice studies have been conducted in the past to assess the organ-specific physiologi-
cal responses to microgravity in short-term and long-term space missions. X.W. Mao et.al.
investigated the effect of a 13-day mission on the cutaneous tissue and found numerous genes
encoding anti-oxidants and extra-cellular matrix (ECM) proteases along with genes responsi-
ble for reactive oxygen species (ROS) generation and gluconeogenesis to be upregulated in the
space flown mice [3]. Spaceflight conditions have also been shown to exert deleterious effects
on the musculoskeletal system. The short-term spaceflight response in the gastrocnemius mus-
cle included decreased PI3-kinase/Akt/mTOR signaling, myogenic cell proliferation, and dif-
ferentiation [4]. The long-term responses in the calf soleus identified changes in expression of
genes involved in maintaining calcium homeostasis, supporting contractile machinery, muscle
development, cell metabolism, and decreasing inflammatory and oxidative stress response [1].
Hepatic lipid metabolism was also evaluated in a 13.5-day mission and early signs of liver
injury was detected in mice [5]. Although these previous studies are starting to reveal the bio-
logical impact of microgravity and/or space radiation on single components of the host, the
overall global response on the host has not yet been determined.
Analysis of complex systems have indicated the difficulty in interpreting collective emer-
gent behaviors occurring as a result of interaction between various components through sepa-
rate analyses of those components [6]. Hence in this study, we provide a systems-level analyses
to assess effects of spaceflight in rodents flown in different habitats for varying time-points, by
evaluating the transcriptional changes in several tissues harvested either post-flight or on-
orbit. Through an established systems biology approach [7–9] we identified a master regulator,
TGF-β1, coordinating systemic responses to microgravity, at multiple biological scales. We
were further able to predict that the global host response between the multiple linked tissues
based on TGF-β1 was driven by circulating microRNAs (miRNAs). MiRNAs are small non-
coding RNA, which can impact a large number of genes, proteins and DNA [10]. Due to the
small size of miRNAs it has been found to be stably circulating throughout the blood, both free
floating and encapsulated in exosomes [11]. We were able to predict that a spaceflight specific
circulating miRNA signature has the potential to drive systemic TGF-β1 response in the host
and display subsequent impact on health, calculated from a biological “health risk score”. The
genes and miRNAs identified from our analyses can be targeted for future research involving
efficient countermeasure design. Our study epitomizes the value of GeneLab data repository,
which can be used not only to retrieve spaceflight data to assess organ/tissue-specific perturba-
tions in molecular signaling networks, but also to establish the foundation of novel hypothe-
sis–based spaceflight research aimed at characterizing the global impact of environmental
stressors at multiple biological scales.
Materials and methods
Ethical approval
This study was conducted in accordance with all ethical standards.
Data from GeneLab platform
All data used for this manuscript were obtained from GeneLab (genelab.nasa.gov). The follow-
ing datasets were used: GLDS-25, -21, -63, -111, -4, -61 and 48. Spaceflight mission and
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
2 / 19
Competing interests: All authors declare no
competing financial interests and conflict of
interest with the data and information in this
manuscript.


# Página 3

experimental details for each dataset such as, the handling of the rodents, tissue processing,
RNA extraction and raw data pertaining to either microarray or RNA-sequencing, can be
found in the GeneLab database. Briefly, we used 7 different murine and rat datasets for our
analysis and examined the following tissues: liver, kidney, adrenal gland, thymus, mammary
gland, skin, and skeletal muscle (soleus, extensor digitorum longus, tibialis anterior, quadri-
ceps, and gastrocnemius) (Fig 1). Specific Details regarding these rodents during spaceflight
are available in the supplemental materials and methods section.
Transcriptome analysis
The available transcriptomic data for tissues from GLDS-25, -21, -63, -111, -4, -61 datasets
were previously performed on different versions of Affymetrix platforms. The transcriptomic
data for the tissues from the GLDS-48, -98, -101, -102, -103, -104, -105 datasets was processed
using RNA-sequencing. Due to the incompatibility of processed data from these different plat-
forms, all datasets were analyzed independently and the processed data was compared across
all tissues. For the GLDS-25 and -4 datasets raw data was background adjusted and quantile
Fig 1. Illustration of methodology and tissues used for analysis. A schematic representation of the process flow in obtaining the omics dataset from GeneLab
database from the tissues used for analyses in this manuscript is shown on the top left. The different tissues have been plotted along an x-axis for the amount of the
time the rodents were in space before sacrificing. The upper panel (blue) represents tissues from rodents on the international space station (ISS) and the lower
panel (yellow) represents the tissues from rodents on space shuttle missions (STS).
https://doi.org/10.1371/journal.pone.0199621.g001
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
3 / 19


# Página 4

normalized done using GenePattern [12]. For GLDS-21, -63, -111, and -61 datasets were back-
ground adjusted and quantile normalized using RMAExpress [13]. The probes from the pre-
processed data were median collapsed using GenePattern. All tissue for the GLDS-48, -98,
-101, -102, -103, -104, -105 data was pre-processed by having Illumina reads trimmed for
sequencing adapters and phred quality score of 20 with Trim-Galore. Alignment to Gencode
Release M15 (GRCm38.p5) transcript sequences and quantification of Gencode comprehen-
sive gene annotations was performed with Kallisto (Kmer length = 31, bootstrap sequences =
100, paired-end, strand specific first read forward) [14]. Read and mapping quality was evalu-
ated with RseQC [15]. Transcript and gene estimated counts and transcripts per million nor-
malizations were performed with Sleuth [16]. All the data for each tissue/dataset was imported
into MultiExperiment Viewer [17] and statistically significant genes was determined either by
t-test with a p-value  0.05 or False Discovery Rate (FDR)  0.05 depending on statistical
stringency needed to produce a reasonable number of genes to take forward for the rest of the
higher-order analysis.
Pathway analysis and subsequent predictions in each tissue were done using the statistically
significant genes with a fold-change 1.2 (or -1.2) comparing Flight Conditions versus
AEM ground controls. Ingenuity Pathway Analysis (IPA) software (Ingenuity Systems) was
used to predict statistically significant activation or inhibition of upstream regulators, canoni-
cal pathways, biofunctions, and toxic functions using activation Z-score statistics ( 2, acti-
vated or  −2, inhibited) [18]. Gene set enrichment analysis (GSEA) was done using the C2,
C5, and Hallmarks gene sets with a FDR  0.05 from the entire list of genes and additional
leading edge analysis was performed as described by Subramanian et. al [19]. All heat maps
and principle component analysis (PCA) plots were generated using packages available
through R (pheatmap for heat maps and scatterplot3d for PCA plots).
Determination of key genes/drivers
We used a previously established unbiased systems biology method to determine key genes/
drivers [7, 9, 20, 21] for each tissue. Briefly, this was done by determining the overlapping
genes involved in the predictions made through IPA’s upstream regulators, canonical path-
ways, biofunctions and GSEA gene sets (C2, C5, and Hallmarks gene sets). The common genes
identified through these statistically significant predictions can be thought of as the central
drivers for the experiment being studied, since the absence of the genes will make these predic-
tions null. To determine the key gene, which has the highest number of connections and can
be thought of as the central hub for the set of key genes, we utilized IPA to define the total
number of connections between all the key genes. Next, we plotted the genes using the radial
plot tool which places the most connected gene in the center of the plot. Previous work involv-
ing similar statistically identified key genes provided experimental validation for this method-
ology using Western blots, qPCR, and other functional assays [7, 20].
MicroRNA (miRNA) predictions and health risk score calculation
Through the use of p-value and activation Z-score statistics in IPA, the top 13 miRNAs impact-
ing the key genes were determined. The upstream regulator tool in IPA was used to determine
the top 10 statistically relevant miRNAs (p-values  1.17 × 10−8) and the prediction of activa-
tion/inhibition of all miRNAs was performed with activation Z-score  2 or  −2 (with corre-
sponding p-values  1.44 × 10−5). Activation Z-score statistics provides the actual functional
activity for the miRNAs, which provides a more meaningful biological representation of the
impact of the miRNAs than p-value alone. The activation Z-score analysis resulted in 4 miR-
NAs, with one miRNA overlapping with the top 10 p-value determined miRNAs. The activity
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
4 / 19


# Página 5

of the remaining 9 miRNAs was determined through activation Z-score statistics in IPA. A full
list of all predicted miRNAs can be found in S1 Table.
The “health risk score” (HRS) was determined by first associating the overall general impact
on health for each miRNA from the literature. Then the HRS was calculated by subtracting the
activation Z-score values (used to determine amount of activation or inhibition) of the miR-
NAs that provided a negative health impact from the miRNAs that provided a positive health
impact. If the overall HRS is positive then it will indicate a beneficial impact on health, while a
negative value will indicate a negative impact on health.
Results and discussion
Description of datasets utilized from GeneLab
GeneLab database was used to obtain microarray and RNA sequencing data pertaining to 12
different mouse datasets (GLDS-25, -21, -111, -4, -61, -48, -98, -101, -102, -103, -104, and
-105) and one rat dataset (GLDS-63 –mammary gland). While microarray expression data was
used for GLDS-25, -21, -63, -111, -4, -61 data sets, NGS (RNA-seq) data was utilized for all tis-
sues related to GLDS-48, -98, -101, -102, -103, -104, -105 data sets. Several tissues were charac-
terized in these datasets: liver, kidney, adrenal gland, thymus, mammary gland, skin, and
skeletal muscle (soleus, extensor digitorum longus, tibialis anterior, quadriceps, and gastrocne-
mius). Fig 1 outlines the various tissues as well as the three different flight systems—five STS,
two ISS and one BION-M1 biosatellite (BF) flights with their respective flight durations. The
detailed description of the age, sex and strain of the animals and type of tissue harvested for
each experiment can be found in the Methods section.
In order to visualize similarity between various animal samples, Principle Component
Analysis (PCA) technique was used for each dataset (S1 Fig). PCA results show that skin sam-
ples from ISS, the majority of the muscle tissue, and mammary glands from STS had clear sep-
aration between the flight and ground Animal Enclosure Module (AEM) controls [22]. When
focusing on the largest experimental dataset and the only RNA-seq dataset (i.e. GLDS-48),
PCA graph shows strong clustering by tissue type, suggesting tissue type is the driving factor
for changes in gene expression (all muscle tissues cluster together—lower left panel, S1 Fig).
Even though, separation between flight and ground AEM is not a predominant feature, we still
ran statistical test to identify genes that were differentially expressed between both experimen-
tal condition for each given tissue and flight condition (see Material and method). We either
used a p-value (p<0.05) cutoff for ISS-flown rodents or a FDR adjusted p values (FDR<0.05)
for STS and Bion-M1- flown samples to reduce the probability of false positive genes (Fig 2).
For each dataset due to the variability in noise and in the number of biological replicates, we
used the lowest stringency for the statistics to produce a reasonable number of significant
genes and the best overlap between each dataset. For example, in the STS-135 liver samples,
STS-70 mammary gland samples, BF SLS and EDL samples, and STS-108 skeletal muscle sam-
ples, a first pass at the data using p-value < 0.05 as a cutoff for significant genes yielded a large
amount of genes: i.e. 17,168 genes, 10,729 genes, 6,801 genes and 9,087 genes respectively. In
contrast, dataset using FDR statistical significant cutoff < 0.05 led to a lower number of genes
suggesting this approach was more stringent statistically and reduced chances of getting false
positives. With this approach, we showed that, overall a higher number of genes were signifi-
cantly upregulated in livers of ISS flown mice compared to the STS flown mice. While in the
ISS- flown mice, the muscles—Soleus, extensor digitorum longus and tibialis anterior, dis-
played the highest number of significant genes, in the STS dataset, the highest number of up-
regulated genes was found in mammary glands (Fig 2). Interestingly the mammary gland tis-
sue was the only tissue from STS flown rats, while all other tissue was from mice. Muscle
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
5 / 19


# Página 6

tissues from Bion-M1 animals displayed a significantly lower number of upregulated genes
than the same tissues from ISS.
Determination of pan-tissue ‘master regulators’ in flight vs ground AEM
studies
In order to gain a comprehensive understanding of how microgravity affected the various
rodent datasets, we first isolated the different factors that varied in the datasets—age, sex, dura-
tion of flight, flight condition (ISS, STS or BF), tissue type and the dataset in question. Using
Ingenuity Pathway Analysis (IPA—QIAGEN Bioinformatics) and the significantly differen-
tially-expressed genes reported in Fig 2, we predicted (based on activation Z-score statistics)
prevalence of change in upstream regulators across all the datasets along with the correspond-
ing pathways that were affected per cluster of upstream regulators (Fig 3A). Upstream regula-
tors are molecular factors (enzymes, kinases, transcriptional regulators, cytokines, growth
factors, etc.) which act as central hubs involved with numerous significantly differentially-
expressed genes [18]. The Prevalence of Change indicates the percentage of datasets in which a
specific upstream regulator (Fig 3A), a canonical pathway (Fig 3B), or a toxicity function (Fig
3C) was predicted to be either activated or inhibited from the list of differentially-expressed
gene for each dataset. We determined factors included in the Prevalence of Change with either
an activation Z-score >0 or <0. For all three analysis we did not find any clustering for activa-
tion or inhibition which correlated with experimental factors (e.g. age, flight duration, sex, tis-
sue type). Among the upstream regulators, TP53 had 100% prevalence of change (Fig 3A) in
response to microgravity, while TGF-β1, UCP1, STAT5B, Ins1, HRAS, MYC, ERK, TNF and
PPARGC1A showed significant activation/inhibition in 87% of the dataset (Fig 3A). Among
the impacted pathways, immune system and inflammation-related and TGF-β1 mediated
pathways were the most prevalent ones (Fig 3A and 3B). Interestingly, mild liver toxicity was
Fig 2. The number of statistically significant genes determined for each tissue and dataset. A bar plot representing the number of significant genes for each
tissue from all datasets either determined by t-test with p-value  0.05 or with FDR  0.05. The tissues are separated by the flight conditions for the rodents with
ISS = International Space Station, STS = Space Shuttle Mission, and BF = Bion. The color-coded bar on the bottom of the plot represents datasets associated with
Flight Duration and Rodent Species.
https://doi.org/10.1371/journal.pone.0199621.g002
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
6 / 19


# Página 7

suggested, with alterations in the pathways “apoptosis of liver cells” and “liver tumor”, along
with changes in the pathways associated with “nephritis” and “cell death of cardiomyocytes”
(Fig 3C).
The predicted factors, TP53, TGF-β1, ERK, HRAS, Ins1, MYC, PPARGC1A, STAT5B, TNF
and UCP1, were affected by microgravity and exerted an overall global impact on all tissues,
and hence were considered as master regulators. Specifically, TP53 and TGF-β1 were predicted
to have modestly high activity in skeletal muscle (STS), liver, kidney and skin of ISS flown
Fig 3. Predicted functions and upstream regulators affected by microgravity. The predicted statistically significant upstream regulators (A), canonical
pathways (B), and toxicity functions (C) determined through IPA from data for each individual tissue/dataset using activation Z-score statistics. Heat map
representation of the activation Z-score values (red = positive activation Z-score for activation and blue = negative activation Z-score for inhibition) were used to
display the data. The prevalence of change (or % of dataset) on the left side of the heat maps represents how common that factor is throughout all datasets/tissues
with the darkest color representing factors with the highest degree in common. Age, sex, tissue type, time in flight, flight conditions, and gene lab dataset reference
is color coded on the top of the heat maps. For the upstream regulators (A) each major cluster of upstream regulators is further analyzed for the major functions it
will impact represented by the Resulting Pathways. D) A bar graph representing predicted activity of TGF-β1 and TP53 through z-score statistics comparing tissue
type, time of flight, and flight conditions, age and sex of the animals.
https://doi.org/10.1371/journal.pone.0199621.g003
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
7 / 19


# Página 8

animals. While the highest expression of TGF-β1 was observed in EDL, TA and ADR from ISS
flown animals TP53 was relatively highly expressed in the STS derived skeletal muscle (Fig
3D). It is interesting to note in this last figure the high correlation between TP53 and TGF-β1
activation levels across the various experiments, suggesting some synergism between both
regulators.
Principal Component Analysis technique was used once more on the predicted activity of
the upstream regulators this time to capture similarity between the different datasets. In con-
trast to PCA of gene expressions (S1 Fig) which could only be done for the same dataset, acti-
vation Z-score for upstream regulators are platform independent and generate a set of vectors
for each dataset that can be plotted simultaneously into one single PCA plot (Fig 4). Doing so,
we clearly identified some tissue separating from all other tissues. The most separated one was
Thymus, which had been reported during the study to show some degree of atrophy with sig-
nificant decrease in leukocyte populations, higher DNA fragmentation, and modulation of
expression of 15 cancer-related genes and 6 T-cell related genes [10]. Mammary gland (MG)
from rats also showed a strong separation at the upstream regulation level. Not as obvious was
Fig 4. Global clustering of the upstream regulators for all tissue types. Principle Component Analysis (PCA) on the upstream regulators for each A) tissue
type, B) age, C) flight duration, and D) flight condition. In A) the Thymus and Mammary Gland (MG) are specifically labeled for clarity in and the muscle data
points are circles in blue. In B) a dark yellow circles all date points representing  16 weeks of age.
https://doi.org/10.1371/journal.pone.0199621.g004
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
8 / 19


# Página 9

the cluster of all muscle tissues which separated based on the second principal component PC2
(vertical axis in Fig 4A). It was also revealed that there is a slight age dependence with older
mice ( 16 weeks) grouping closer together (Fig 4B). This indicates that the overall age of the
rodents used has a slight impact on the systemic biological response to microgravity. Changes
in predicted activity of the upstream regulators on all the tissues seem to be sex, flight condi-
tions and duration of flight independent (Fig 4C and 4D).
Focusing on the muscle tissue cluster identified in Fig 4A (all showing up inside or near the
PC2 positive half), one can note that individual tissue samples have a large spread along the
first component in the PCA plot (PC1 –horizontal axis), suggesting different biological pro-
cesses are at play depending on the muscle type along this component. Performing hierarchical
clustering analysis for the activation Z-scores of the upstream regulators in muscle tissue
alone, muscle types naturally divided into two sub groups: Gastrocnemius and Soleus (“Group
1”, green-bordered box in Fig 5) which displayed similar patterns of predicted altered activity
for the majority of the upstream regulators, and Extensor Digitorum Longus, Quadriceps,
Tibialis Anterior and Skeletal Muscle (“Group 2”) which consistently had opposite patterns
from Group 1 (Fig 5). The two groups of muscle tissues displayed significant differences in the
level of predicted activity of crucial signaling genes such as TNF, TGF-β1, TGFα, p38MAPK,
ERK, and IL1β. We also found that the predicted activity of PPARGC-1α was highly downre-
gulated in Group 1 muscle and upregulated in all but one of the types in Group 2. This could
be due to potential compensatory mechanisms occurring in certain muscle types of the muscu-
loskeletal system as a result of microgravity exposure. By looking at the anatomy of a mouse
muscle (S2 Fig), one could interpret these results as an unloading of the muscles in Group 1
not supporting the weight of the animal in microgravity anymore, while muscles in Group 2
would see an increased loading due to pull and grabbing to the cage. Interestingly, micrograv-
ity responsive changes in immune-related pathway regulation were observed to be the most
prevalent in the muscle Group 1.
Determination of key microgravity responsive genes and the most
connected gene(s)
To determine key genes that impacted the physiological and biochemical processes in rodents
exposed to microgravity, we used a system biology approach we previously introduced [7, 9,
21]. Briefly, this method combines statistical tools contained in Gene Set Enrichment Analysis
[19] and IPA (detailed description available in methods) to identify the “key genes” driving the
observed differentially-expressed genes. We hypothesize, these “key genes” are driving the bio-
logical response to spaceflight conditions. To determine if any commonality exists between the
sets of key genes for each dataset, we diagrammed connectivity between each dataset to show
that a significant number of key genes are shared (Fig 6A and S3 Fig). The specific connections
between the key genes associated with each dataset can be thought of central hubs (Fig 6A).
The details for the rest of the key genes for each dataset can be found in S3 Fig. These key
genes that are shared between the datasets are hypothesized to have the highest global impact
in the host affecting multiple tissues, in response to microgravity.
Note that Fig 6A identified CDKN1A, IL6, ICAM and MAPK11 as key genes, which have
common immune pathways and have previously been shown to be impacted by microgravity
across various tissues [23, 24]. Similarly, Thymus isolated from STS-118 mice also showed
downregulation of expression in several of the signaling nodes such as CCND1, TGF-β1, and
MYC as previously shown [23]. The most connected key gene/driver in all tissues (for studies
comparing Flight versus ground AEM) was TGF-β1 (Fig 6B). Transforming growth factor beta
(TGF-β) is a pleiotropic cytokine; known to play a context specific role in sustaining tissue
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
9 / 19


# Página 10

Fig 5. Predicted upstream regulators for muscle tissue affected by microgravity. The statistically significant predicted upstream regulators
determined by IPA from data for each individual tissue/dataset using activation Z-score statistics. Heat map representation with hierarchical
clustering of the activation Z-score values (red = positive activation Z-score for activation and blue = negative activation Z-score for
inhibition) were used to display the data. The prevalence of change (or % of dataset) on the left side of the heat maps represents how common
that factor is throughout all datasets/tissues with the darkest color representing factors with the highest degree in common. Age, sex, tissue
type, time in flight, flight conditions, and gene lab dataset reference is color coded on the top of the heat maps. Each major cluster of
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
10 / 19


# Página 11

homeostasis predominantly via transcriptional regulation of genes involved in differentiation,
cell motility, proliferation, cell survival along with regulating immune responses during
homeostasis and infection [25].
In an in vitro study by Blaber et al. [26], the effects of 15 days of microgravity were assessed
on early lineage commitment of mouse embryonic stem cells (mESCs) using the embryoid
body (EB) model of tissue differentiation. Although no significant changes were observed in
TRP53 in EBs differentiated in microgravity, a few apoptosis related genes such as BCL2,
CUL9, FADD and CASP9 were found to be upregulated in EBs grown at 1g relative to the
undifferentiated mESCs. A few p53 target genes such as STAT1, JUN, and EGR1 were also sig-
nificantly down regulated in microgravity-differentiated EBs, relative to 1g control EBs [26].
We compared the overlapping key genes (Fig 6 and S3 Fig) from our analysis (comparing gene
expression values) with the genes identified in the microgravity induced EB differentiation
and categorized the genes based on their functions (S4 Fig). The full list of these genes and
associated categories/pathways are found in Blaber et al [26]. Although in vitro conditions may
not accurately replicate those in vivo, we found similar trends in regulation of several genes in
the functional categories of ‘apoptosis’ and ‘cell cycle regulation’. BIRC5, CASP9, SFN and TNF
involved in apoptosis and CDC25Aand KRAS involved in cell cycle regulation varied consis-
tently between the two studies. The expression of key genes derived from our analysis of in
upstream regulators is further analyzed for the major functions it will impact represented by the Resulting Pathways. The green
box represents the division between group 1 of muscles (consisting of Soleus and Gastrocnemius) and group 2 (consisting of Extensor
Digitorum Longus, Tibialis Anterior, and Quadriceps) having an overall opposite regulation for most of the upstream regulators.
https://doi.org/10.1371/journal.pone.0199621.g005
Fig 6. Key genes and master regulators driving pan-tissue microgravity response. A) A connectivity network for each set of key genes determined
independently for each tissue/dataset. The key genes associated with each individual tissue/dataset are represented by different background colors as indicated in
the legend. Overlapping key genes (blue font) between each tissue type is represented by background colors and is considered as central nodes. The cluster of key
genes for each dataset that are not connected are grouped and shown as one circle. Details for each of these clusters of key genes for each dataset can be seen in S2
Fig. The color of the gene represents whether the gene is upregulated (red) or downregulated (green) with the shade signifying the degree of regulation. The
different line colors represent the predicted effect of each gene on each other. B) A radial plot of all connections between all key genes with the most connected
gene displayed in the middle (TGFβ1). The key genes with direct connections to TGFβ1 are shown with all other connections shown with faded color.
https://doi.org/10.1371/journal.pone.0199621.g006
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
11 / 19


# Página 12

vivo flight data, such as IL6, STAT1, PPARG/A and CCND1/2 (Fig 6) also varied consistently in
our study and in Blaber et al. study, suggesting that in vivo pan-tissue analysis of gene expres-
sion can be used to confirm knowledge about spaceflight related biological effects from in vitro
studies.
Delineating the spaceflight-induced circulating miRNA signature and
global health risk assessment
Examination of a common circulating factor that could connect the TGF-β1 led myriad signal-
ing pathways coordinating systemic response to spaceflight, was warranted given the influence
of microRNAs (miRNAs) on TGF-β and p53. MiRNAs are endogenous small noncoding
RNAs that each can target hundreds of mRNAs (also protein and DNA) and function as post-
transcriptional modulators of gene expression, leading to dysregulation of protein expression
and/or mRNA levels [11, 27, 28]. Simulated microgravity study detected elevated levels of
miR-223 with decreased proliferation of Hepa1-6 cells [29]. Consistent with this report, our
analyses also predicted miR-223 among the top 10 spaceflight-induced miRNAs, which poten-
tially could result in down regulation of several key genes such as ICAM1, IGF1R and upregula-
tion of TLR4/7, LPL and CCR3 etc. (Fig 7A and S1 Table). In another report, modeled
microgravity was found to cause significant overexpression of miR-34a in human lymphoblas-
toid cells [30]. Our analyses also detected miR-34 to be significantly activated in response to
spaceflight, which is involved in potential downregulation of several key genes such as
CCND2/1,MYC, CDC25a and LEF1 (Fig 7B). Compared to normal gravity, in-silico analysis of
stimulated human leukocytes showed that brief exposure to spaceflight onboard ISS caused
suppression of miR-21 [31]. The same miRNA was predicted in our analyses to cause downre-
gulation of critical key genes such as TGF-β1, PPARA, PTEN, CDC25a, NRLC5, ICAM1 and
ILIB and activation of COL1A1 (Fig 7A). Using the list of predicted miRNAs a biological
Health Risk Score (HRS) was calculated. We have previously used a similar method to calculate
a Cancer Risk Score [11]. This HRS provides us with a comprehensive idea about how a group
of spaceflight responsive miRNAs could weigh on overall health, positive for lower health risk
and negative for higher health risk (Fig 7C). miR-25 and miR-17-5p which have positive health
risk showed a predicted inhibition (blue color) leading to a negative impact on health. All the
other predicted miRNAs are activated and are known to have a negative impact on health [32–
34]. MiR-26a-5p is shown in Fig 7C to have both positive and negative impact on heath. Over-
all, these spaceflight studies suggest microgravity onboard ISS, STS and Bion-M1 have a strong
negative impact on rodent health, with an HRS score of -12.79.
Fig 7D shows the direct interaction of the majority of the predicted microgravity associated
miRNAs with the master regulators TGF-β and p53. For example, TGF-β and p53 have been
previously implicated in biogenesis of miRNAs including miR-215, which may decrease cell
division [28]. TGF-β can function to increase maturation of miR-21, which was shown to
inhibit PTEN and Sprouty 1, the crucial negative regulators of the Akt and Ras/MAPK path-
ways, thus leading to tumor progression [35, 36]. MiR-34 has been found to inhibit TP53
through directly targeting its mRNA [37].
Conclusions
The analyses presented here demonstrate the utility of publicly available repositories like
NASA’s GeneLab to generate hypotheses on the biological impact of microgravity. We found
that, irrespective of rodent type, age, sex, flight condition and time of flight, several master reg-
ulators coordinated major systemic responses towards microgravity. TP53, TGF-β1 and
immune associated signaling were identified to be the most prevalent pan-tissue signaling
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
12 / 19


# Página 13

nodes activated in response to microgravity along with TGF-β1 being the most connected gene
across all datasets.
Transforming growth factor beta (TGF-β) is a pleiotropic cytokine, belonging to a family,
which consists of 33 members, including the activins, inhibins, bone morphogenetic proteins
(BMPs) and growth and differentiation factors. TGF-β is known to play a context-specific role
Fig 7. A microgravity associated circulating miRNA signature. A) A radial plot showing the top 10 predicted miRNAs with p-values  1.17 × 10−8 determined
from all key genes and the key genes directly related the miRNAs. B) All statistical significant miRNAs predicted from all key genes with activation Z-score  2 or
 -2 and the corresponding key genes associated with these miRNAs. The predicted activity of each miRNA (blue = inhibition and orange = activation) was
determined through activation Z-score statistics through IPA. C) A graphical representation of the health risk score (HRS) illustrating how each miRNA
contributes to the calculation. The outline for each miRNA represents if the miRNA has a negative impact on health (black), positive impact on health (olive), and
has both positive and negative impact on health (grey). D) Radial plot connecting TGFβ1 with p53 and all miRNAs. Through activation Z-score statistics in IPA it
was determined that p53 will be activated due to the impact of TGFβ1 and the miRNAs.
https://doi.org/10.1371/journal.pone.0199621.g007
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
13 / 19


# Página 14

in sustaining tissue homeostasis predominantly via transcriptional regulation of genes
involved in differentiation, cell motility, proliferation, and cell survival along with regulating
immune responses during homeostasis and infection. Several previous reports have indicated
the modulation of TGF-β gene expression with microgravity. For example, reduction in gravi-
tational force was found to diminish TGF-β expression and apoptosis with higher carcinoem-
bryonic antigen expression in 3D human colorectal carcinoma cells, as compared to 3D
cultures in unit gravity [38]. In another study, differential regulation of blood vessel growth by
TGF-β using basic fibroblast growth factor was identified in modeled microgravity with induc-
tion of early and late apoptosis, extracellular matrix proteins, endothelin-1 and TGF-β1 expres-
sion [39]. Bone development involves dynamic remodeling involving gravity-regulated
mechanical stimulation for conservation of mineral content and structure. TGF-β has been
implicated to function as an autocrine and paracrine regulator of bone formation. Human
fetal osteoblastic (hFOB) cells grown in space have been shown to exhibit significantly reduced
TGF-β1 and TGF-β2 transcript levels, 24 hours post-flight [40]. Among in vivo studies, both
short term and long-term flight data report changes in TGF-β mRNA levels in response to
weightlessness and microgravity. During an 11-day space mission, rat skeletal muscle exhib-
ited reduced TGF-β expression [41], and a 91-day mission aboard the ISS revealed lower TGF-
β expression in colonic tissue, systemic lymph node, inguinal and brachial lymph nodes [42].
Microgravity is one possible cause changing TGF-β expression levels. However, ionizing
radiation found in space can also have impact on TGF-β expression levels and the overall biol-
ogy of organisms in space [43, 44]. For example, upregulation of TGF-β following ionizing
radiation may improve DNA repair [45, 46] but it can also elicit a stem-cell self-renewal signal-
ing in breast cells which has been correlated to increased breast cancer risk in young women
exposed to ionizing radiation [47]. In addition, our results show that the space environment is
globally down-regulating TGF-β1 in the rodent tissue (Figs 6 and 7), which may contribute to
additional DNA damage to the host due to HZE irradiation but which may also lower certain
cancer risk. These results also suggest that microgravity is the predominant factor affecting
TGF-β, leading to lower expression levels in contradiction from what one expects from expo-
sure to ionizing radiation.
TGF-β signaling has also been known to crosstalk with the second key regulator found in
this study: i.e. p53 [28]. p53 is a transcription factor and in response to genotoxic stress, DNA
damage, oncogene activation, and hypoxia, it is recruited to specific sites in chromatin, pro-
moting transcription of apoptosis related genes [48]. Again, the relationship with TGF-β is
complex as a report showed that inactivation of p53 can also alter TGF-β signaling, which iron-
ically displayed both tumor-suppressive and pro-oncogenic functions [49].
Finally, we hypothesized that the global systemic response to microgravity in rodents,
driven by TGF-β1 is arbitrated by a circulating miRNA signature consisting of thirteen miR-
NAs predicted from the key genes. Using the miRNAs functional state and impact on health,
we calculated a theoretical “health risk score”. The Health Risk Score is based on known associ-
ation of this miRNA with reported health outcomes, circumventing the challenging interpreta-
tion for health effects resulting from the complex interaction of TGF-β/p53/Immune signaling
discussed previously. With this novel approach, we predict that short and long-term space mis-
sions exert a strong negative impact on rodent health. Suprisingly the miRNAs that we predict
to drive microgravity biological response have been reported to be involved with simulated
microgravitry experiments previously reported by other investigators. For example, modeled
microgravity-triggered miRNAs have been identified in human peripheral blood lymphocytes
(PBL) [27, 50], human leukocytes [31], human lymphoblastoid cells [30], and in murine hepa-
toma cell line [29]. In these studies, miR-223 has been found to regulate cell cycle progression
by targeting E2F1, Fbxw7/Cdc4, IGF1R, Cdk2 and TOX and overexpression of the miRNA
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
14 / 19


# Página 15

resulting in suppression of c-Myc expression [51]. Human peripheral blood lymphocytes
(PBLs) cultured for 24 h in microgravity with respect to 1 g, revealed dysregulation of 42 miR-
NAs, of which miR-34a-5p, miR-34b-5p, and let-7i-3p were found to be in common with our
results. Most of the identified miRNAs were correlated with controlling immune-related (TCR
signaling, adaptive and innate immune signaling and cytokine signaling), apoptosis and cell
proliferation related gene expression.
Several clinical analyses have shown that circulating miRNAs are stable in serum and
plasma, are water-soluble, and are easily detectable. Given their modulation of expression pro-
file based on physiological and pathological conditions, their role as therapeutics biomarkers is
being investigated [52]. Recent reports on regulation of cell-cell communication by exosomes
have led to researchers to investigate further the role of exosome-derived miRNA that can
function in the target cells in diverse biological processes [53]. Other than microgravity, acute
and chronic levels of ionizing radiation pose threats to astronaut health in space missions [54].
Radiation induced DNA damage has been studied in detail and changes in miRNA expression
was detected both in vivo and in vitro and based on type of cell, type of radiation and repair
time, miRNA levels were found to differentially synchronize p53 activity [55, 56]. Hence, cell-
free, circulating miRNA may be a useful minimally non-invasive biomarker for the detection
of space related health risk, also to monitor progress of the symptoms, and subsequent
response to therapeutic countermeasures.
In conclusion, the current study demonstrates the value of repository data in generating
novel hypotheses through in-silico analysis. This study revealed microgravity-induced critical
genes, signaling pathways and circulating miRNA signatures that may be leveraged for identi-
fication of space related health risks biomarkers and in the design of countermeasures for
long-term manned missions.
Supporting information
S1 Fig. Global clustering of Flight versus AEM ground controls for each individual dataset.
Principle component analysis (PCA) determined from all probes for each dataset comparing
Flight versus AEM ground controls.
(TIF)
S2 Fig. Schematic of the ventral and medial muscles in a mouse. A schematic depicting the
different groups of muscles in the ventral and medial muscles for a mouse. Muscle Group 1
represents the posterior muscles in the lower leg, including the Gastrocnemius and Soleus, and
comprises ~20% of the leg muscle mass. Muscle Group 2a represents the anterior muscles of
the lower leg, including tibialis anterior and extensor digitorum longus, and comprises of
~10% of the leg muscle mass. Muscle Group 2b represents the medial muscles in the upper leg
and comprises of 25% of leg muscle mass. Muscle Group 2c represents the anterior muscles of
the upper leg, including the Quadriceps, and comprises ~20% of the leg muscle mass. Muscle
Group 1 is the cluster represented as Group 1 in Fig 5 and all other muscle groups fall in
Group 2 in Fig 5.
(TIF)
S3 Fig. Detailed network of all key genes and connections between each dataset. A detailed
network of the key genes found in Fig 6A for each dataset. Background color for each dataset
is provided for each set of key genes. The overlapping key genes are similar to Fig 6A. Each
gene is represented by a symbol indicating what type of molecule it is. The color of the gene
represents whether the gene is upregulated (red) or downregulated (green) signifying the
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
15 / 19


# Página 16

degree of regulation. The different line colors represent the predicted effect of each gene on
each other.
(TIF)
S4 Fig. The overlap of all key genes/drivers compared to genes from existing literature. All
key genes we determined from all tissues/datasets were compared to the genes impacted by
microgravity discussed by Blaber et al [26]. A box plot representation of the fold-change values
of the overlapping genes is displayed with our analysis (black) and the values found by Blaber
et. al. (red) [26]. The background colors represent the functional category for each group of
genes as presented in the Blaber et. al. manuscript [26].
(TIF)
S1 Table. Predicted microRNAs (miRNAs) from the all key genes determined through
Ingenuity Pathway Analysis (IPA). Activation Z-scores > 0 will predict the miRNA is acti-
vated and activation Z-scores < 0 will predict the miRNAs are inhibited. The bold miRNAs
are the top miRNAs which were used in our analysis.
(PDF)
S1 Materials and Methods. Data from GeneLab platform.
(PDF)
Acknowledgments
The authors would like to acknowledge the NASA GeneLab Data management team member
Samrawit Gebre for her help in curating and managing the experimental database.
Author Contributions
Conceptualization: Afshin Beheshti.
Formal analysis: Afshin Beheshti, Homer Fogle.
Investigation: Afshin Beheshti.
Methodology: Afshin Beheshti.
Supervision: Afshin Beheshti.
Visualization: Afshin Beheshti.
Writing – original draft: Afshin Beheshti, Shayoni Ray.
Writing – review & editing: Afshin Beheshti, Daniel Berrios, Sylvain V. Costes.
References
1.
Gambara G, Salanova M, Ciciliot S, Furlan S, Gutsmann M, Schiffl G, et al. Gene Expression Profiling
in Slow-Type Calf Soleus Muscle of 30 Days Space-Flown Mice. PLoS One. 2017; 12(1):e0169314.
https://doi.org/10.1371/journal.pone.0169314 PMID: 28076365.
2.
Andreev-Andrievskiy A, Popova A, Boyle R, Alberts J, Shenkman B, Vinogradova O, et al. Mice in Bion-
M 1 space mission: training and selection. PLoS One. 2014; 9(8):e104830. https://doi.org/10.1371/
journal.pone.0104830 PMID: 25133741.
3.
Mao XW, Pecaut MJ, Stodieck LS, Ferguson VL, Bateman TA, Bouxsein ML, et al. Biological and meta-
bolic response in STS-135 space-flown mouse skin. Free Radic Res. 2014; 48(8):890–7. https://doi.
org/10.3109/10715762.2014.920086 PMID: 24796731.
4.
Allen DL, Bandstra ER, Harrison BC, Thorng S, Stodieck LS, Kostenuik PJ, et al. Effects of spaceflight
on murine skeletal muscle gene expression. J Appl Physiol (1985). 2009; 106(2):582–95. https://doi.
org/10.1152/japplphysiol.90780.2008 PMID: 19074574.
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
16 / 19


# Página 17

5.
Jonscher KR, Alfonso-Garcia A, Suhalim JL, Orlicky DJ, Potma EO, Ferguson VL, et al. Spaceflight
Activates Lipotoxic Pathways in Mouse Liver. PLoS One. 2016; 11(4):e0152877. https://doi.org/10.
1371/journal.pone.0152877 PMID: 27097220.
6.
Ray S, Yuan D, Dhulekar N, Oztan B, Yener B, Larsen M. Cell-based multi-parametric model of cleft
progression during submandibular salivary gland branching morphogenesis. PLoS Comput Biol. 2013;
9(11):e1003319. https://doi.org/10.1371/journal.pcbi.1003319 PMID: 24277996.
7.
Ravi D, Beheshti A, Abermil N, Passero F, Sharma J, Coyle M, et al. Proteasomal Inhibition by Ixazomib
Induces CHK1 and MYC-Dependent Cell Death in T-cell and Hodgkin Lymphoma. Cancer Res. 2016;
76(11):3319–31. https://doi.org/10.1158/0008-5472.CAN-15-2477 PMID: 26988986.
8.
Wage J, Ma L, Peluso M, Lamont C, Evens AM, Hahnfeldt P, et al. Proton irradiation impacts age-driven
modulations of cancer progression influenced by immune system transcriptome modifications from
splenic tissue. J Radiat Res. 2015; 56(5):792–803. https://doi.org/10.1093/jrr/rrv043 PMID: 26253138.
9.
Beheshti A, Cekanaviciute E, Smith DJ, Costes SV. Global transcriptomic analysis suggests carbon
dioxide as an environmental stressor in spaceflight: A systems biology GeneLab case study. Sci Rep.
2018; 8(1):4191. https://doi.org/10.1038/s41598-018-22613-1 PMID: 29520055.
10.
Jasinski-Bergner S, Mandelboim O, Seliger B. The role of microRNAs in the control of innate immune
response in cancer. J Natl Cancer Inst. 2014; 106(10). https://doi.org/10.1093/jnci/dju257 PMID:
25217579.
11.
Beheshti A, Vanderburg C, McDonald JT, Ramkumar C, Kadungure T, Zhang H, et al. A Circulating
microRNA Signature Predicts Age-Based Development of Lymphoma. PLoS One. 2017; 12(1):
e0170521. https://doi.org/10.1371/journal.pone.0170521 PMID: 28107482.
12.
Reich M, Liefeld T, Gould J, Lerner J, Tamayo P, Mesirov JP. GenePattern 2.0. Nat Genet. 2006;
38(5):500–1. https://doi.org/10.1038/ng0506-500 PMID: 16642009.
13.
Bolstad BM, Irizarry RA, Astrand M, Speed TP. A comparison of normalization methods for high density
oligonucleotide array data based on variance and bias. Bioinformatics. 2003; 19(2):185–93. PMID:
12538238.
14.
Bray NL, Pimentel H, Melsted P, Pachter L. Near-optimal probabilistic RNA-seq quantification. Nat Bio-
technol. 2016; 34(5):525–7. https://doi.org/10.1038/nbt.3519 PMID: 27043002.
15.
Wang L, Wang S, Li W. RSeQC: quality control of RNA-seq experiments. Bioinformatics. 2012;
28(16):2184–5. https://doi.org/10.1093/bioinformatics/bts356 PMID: 22743226.
16.
Pimentel H, Bray NL, Puente S, Melsted P, Pachter L. Differential analysis of RNA-seq incorporating
quantification uncertainty. Nat Methods. 2017; 14(7):687–90. https://doi.org/10.1038/nmeth.4324
PMID: 28581496.
17.
Saeed AI, Bhagabati NK, Braisted JC, Liang W, Sharov V, Howe EA, et al. TM4 microarray software
suite. Methods Enzymol. 2006; 411:134–93. https://doi.org/10.1016/S0076-6879(06)11009-5 PMID:
16939790.
18.
Kramer A, Green J, Pollard J Jr, Tugendreich S. Causal analysis approaches in Ingenuity Pathway
Analysis. Bioinformatics. 2014; 30(4):523–30. https://doi.org/10.1093/bioinformatics/btt703 PMID:
24336805.
19.
Subramanian A, Tamayo P, Mootha VK, Mukherjee S, Ebert BL, Gillette MA, et al. Gene set enrichment
analysis: a knowledge-based approach for interpreting genome-wide expression profiles. Proc Natl
Acad Sci U S A. 2005; 102(43):15545–50. https://doi.org/10.1073/pnas.0506580102 PMID: 16199517.
20.
Beheshti A, Benzekry S, McDonald JT, Ma L, Peluso M, Hahnfeldt P, et al. Host age is a systemic regu-
lator of gene expression impacting cancer progression. Cancer Res. 2015; 75(6):1134–43. https://doi.
org/10.1158/0008-5472.CAN-14-1053 PMID: 25732382.
21.
Beheshti A, Wage J, McDonald JT, Lamont C, Peluso M, Hahnfeldt P, et al. Tumor-host signaling inter-
action reveals a systemic, age-dependent splenic immune influence on tumor development. Oncotar-
get. 2015; 6(34):35419–32. https://doi.org/10.18632/oncotarget.6214 PMID: 26497558.
22.
Moyer EL, Dumars PM, Sun GS, Martin KJ, Heathcote DG, Boyle RD, et al. Evaluation of rodent space-
flight in the NASA animal enclosure module for an extended operational period (up to 35 days). NPJ
Microgravity. 2016; 2:16002. https://doi.org/10.1038/npjmgrav.2016.2 PMID: 28725722.
23.
Gridley DS, Mao XW, Stodieck LS, Ferguson VL, Bateman TA, Moldovan M, et al. Changes in mouse
thymus and spleen after return from the STS-135 mission in space. PLoS One. 2013; 8(9):e75097.
https://doi.org/10.1371/journal.pone.0075097 PMID: 24069384.
24.
Gridley DS, Slater JM, Luo-Owen X, Rizvi A, Chapes SK, Stodieck LS, et al. Spaceflight effects on T
lymphocyte distribution, function and gene expression. J Appl Physiol (1985). 2009; 106(1):194–202.
https://doi.org/10.1152/japplphysiol.91126.2008 PMID: 18988762.
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
17 / 19


# Página 18

25.
Worthington JJ, Fenton TM, Czajkowska BI, Klementowicz JE, Travis MA. Regulation of TGFbeta
in the immune system: an emerging role for integrins and dendritic cells. Immunobiology. 2012;
217(12):1259–65. https://doi.org/10.1016/j.imbio.2012.06.009 PMID: 22902140.
26.
Blaber EA, Finkelstein H, Dvorochkin N, Sato KY, Yousuf R, Burns BP, et al. Microgravity Reduces
the Differentiation and Regenerative Potential of Embryonic Stem Cells. Stem Cells Dev. 2015;
24(22):2605–21. https://doi.org/10.1089/scd.2015.0218 PMID: 26414276.
27.
Girardi C, De Pitta C, Casara S, Calura E, Romualdi C, Celotti L, et al. Integration analysis of microRNA
and mRNA expression profiles in human peripheral blood lymphocytes cultured in modeled micrograv-
ity. Biomed Res Int. 2014; 2014:296747. https://doi.org/10.1155/2014/296747 PMID: 25045661.
28.
Elston R, Inman GJ. Crosstalk between p53 and TGF-beta Signalling. J Signal Transduct. 2012;
2012:294097. https://doi.org/10.1155/2012/294097 PMID: 22545213.
29.
Chen Y, Xu J, Yang C, Zhang H, Wu F, Chen J, et al. Upregulation of miR-223 in the rat liver inhibits pro-
liferation of hepatocytes under simulated microgravity. Exp Mol Med. 2017; 49(6):e348. https://doi.org/
10.1038/emm.2017.80 PMID: 28642576.
30.
Mangala LS, Zhang Y, He Z, Emami K, Ramesh GT, Story M, et al. Effects of simulated microgravity on
expression profile of microRNA in human lymphoblastoid cells. J Biol Chem. 2011; 286(37):32483–90.
https://doi.org/10.1074/jbc.M111.267765 PMID: 21775437.
31.
Hughes-Fulford M, Chang TT, Martinez EM, Li CF. Spaceflight alters expression of microRNA during T-
cell activation. FASEB J. 2015; 29(12):4893–900. https://doi.org/10.1096/fj.15-277392 PMID:
26276131.
32.
Cloonan N, Brown MK, Steptoe AL, Wani S, Chan WL, Forrest AR, et al. The miR-17-5p microRNA is a
key regulator of the G1/S phase cell cycle transition. Genome Biol. 2008; 9(8):R127. https://doi.org/10.
1186/gb-2008-9-8-r127 PMID: 18700987.
33.
Komatsu S, Ichikawa D, Hirajima S, Kawaguchi T, Miyamae M, Okajima W, et al. Plasma microRNA
profiles: identification of miR-25 as a novel diagnostic and monitoring biomarker in oesophageal squa-
mous cell carcinoma. Br J Cancer. 2014; 111(8):1614–24. https://doi.org/10.1038/bjc.2014.451 PMID:
25117812.
34.
Wahlquist C, Jeong D, Rojas-Munoz A, Kho C, Lee A, Mitsuyama S, et al. Inhibition of miR-25 improves
cardiac contractility in the failing heart. Nature. 2014; 508(7497):531–5. https://doi.org/10.1038/
nature13073 PMID: 24670661.
35.
Polytarchou C, Iliopoulos D, Hatziapostolou M, Kottakis F, Maroulakou I, Struhl K, et al. Akt2 regulates
all Akt isoforms and promotes resistance to hypoxia through induction of miR-21 upon oxygen depriva-
tion. Cancer Res. 2011; 71(13):4720–31. https://doi.org/10.1158/0008-5472.CAN-11-0365 PMID:
21555366.
36.
Meng F, Henson R, Wehbe-Janek H, Ghoshal K, Jacob ST, Patel T. MicroRNA-21 regulates expres-
sion of the PTEN tumor suppressor gene in human hepatocellular cancer. Gastroenterology. 2007;
133(2):647–58. https://doi.org/10.1053/j.gastro.2007.05.022 PMID: 17681183.
37.
Navarro F, Lieberman J. miR-34 and p53: New Insights into a Complex Functional Relationship. PLoS
One. 2015; 10(7):e0132767. https://doi.org/10.1371/journal.pone.0132767 PMID: 26177460.
38.
Jessup JM, Frantz M, Sonmez-Alpan E, Locker J, Skena K, Waller H, et al. Microgravity culture reduces
apoptosis and increases the differentiation of a human colorectal carcinoma cell line. In Vitro Cell Dev
Biol Anim. 2000; 36(6):367–73. PMID: 10949995.
39.
Ulbrich C, Westphal K, Baatout S, Wehland M, Bauer J, Flick B, et al. Effects of basic fibroblast growth
factor on endothelial cells under conditions of simulated microgravity. J Cell Biochem. 2008; 104
(4):1324–41. https://doi.org/10.1002/jcb.21710 PMID: 18253936.
40.
Harris SA, Zhang M, Kidder LS, Evans GL, Spelsberg TC, Turner RT. Effects of orbital spaceflight on
human osteoblastic cell physiology and gene expression. Bone. 2000; 26(4):325–31. https://doi.org/10.
1016/S8756-3282(00)00234-9 PMID: 10719274.
41.
Westerlind KC, Turner RT. The skeletal effects of spaceflight in growing rats: tissue-specific alterations
in mRNA levels for TGF-beta. J Bone Miner Res. 1995; 10(6):843–8. https://doi.org/10.1002/jbmr.
5650100603 PMID: 7572306.
42.
McCarville JL, Clarke ST, Shastri P, Liu Y, Kalmokoff M, Brooks SP, et al. Spaceflight influences both
mucosal and peripheral cytokine production in PTN-Tg and wild type mice. PLoS One. 2013; 8(7):
e68961. https://doi.org/10.1371/journal.pone.0068961 PMID: 23874826.
43.
Barcellos-Hoff MH, Blakely EA, Burma S, Fornace AJ Jr, Gerson S, Hlatky L, et al. Concepts and chal-
lenges in cancer risk prediction for the space radiation environment. Life Sci Space Res (Amst). 2015;
6:92–103. https://doi.org/10.1016/j.lssr.2015.07.006 PMID: 26256633.
44.
Durante M, Cucinotta FA. Heavy ion carcinogenesis and human space exploration. Nat Rev Cancer.
2008; 8(6):465–72. https://doi.org/10.1038/nrc2391 PMID: 18451812.
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
18 / 19


# Página 19

45.
Barcellos-Hoff MH, Cucinotta FA. New tricks for an old fox: impact of TGFbeta on the DNA damage
response and genomic stability. Sci Signal. 2014; 7(341):re5. https://doi.org/10.1126/scisignal.2005474
PMID: 25185158.
46.
Kim MR, Lee J, An YS, Jin YB, Park IC, Chung E, et al. TGFbeta1 protects cells from gamma-IR by
enhancing the activity of the NHEJ repair pathway. Mol Cancer Res. 2015; 13(2):319–29. https://doi.
org/10.1158/1541-7786.MCR-14-0098-T PMID: 25319009.
47.
Tang J, Fernandez-Garcia I, Vijayakumar S, Martinez-Ruis H, Illa-Bochaca I, Nguyen DH, et al. Irradia-
tion of juvenile, but not adult, mammary gland increases stem cell self-renewal and estrogen receptor
negative tumors. Stem cells. 2014; 32(3):649–61. https://doi.org/10.1002/stem.1533 PMID: 24038768.
48.
Oda E, Ohki R, Murasawa H, Nemoto J, Shibue T, Yamashita T, et al. Noxa, a BH3-only member of the
Bcl-2 family and candidate mediator of p53-induced apoptosis. Science. 2000; 288(5468):1053–8.
PMID: 10807576.
49.
Nguyen DH, Oketch-Rabah HA, Illa-Bochaca I, Geyer FC, Reis-Filho JS, Mao JH, et al. Radiation acts
on the microenvironment to affect breast carcinogenesis by distinct mechanisms that decrease cancer
latency and affect tumor type. Cancer Cell. 2011; 19(5):640–51. https://doi.org/10.1016/j.ccr.2011.03.
011 PMID: 21575864.
50.
Girardi C, De Pitta C, Casara S, Sales G, Lanfranchi G, Celotti L, et al. Analysis of miRNA and mRNA
expression profiles highlights alterations in ionizing radiation response of human lymphocytes under
modeled microgravity. PLoS One. 2012; 7(2):e31293. https://doi.org/10.1371/journal.pone.0031293
PMID: 22347458.
51.
McGirt LY, Adams CM, Baerenwald DA, Zwerner JP, Zic JA, Eischen CM. miR-223 regulates cell
growth and targets proto-oncogenes in mycosis fungoides/cutaneous T-cell lymphoma. J Invest Derma-
tol. 2014; 134(4):1101–7. https://doi.org/10.1038/jid.2013.461 PMID: 24304814.
52.
Wang WT, Chen YQ. Circulating miRNAs in cancer: from detection to therapy. J Hematol Oncol. 2014;
7:86. https://doi.org/10.1186/s13045-014-0086-0 PMID: 25476853.
53.
Valadi H, Ekstrom K, Bossios A, Sjostrand M, Lee JJ, Lotvall JO. Exosome-mediated transfer of
mRNAs and microRNAs is a novel mechanism of genetic exchange between cells. Nat Cell Biol. 2007;
9(6):654–9. https://doi.org/10.1038/ncb1596 PMID: 17486113.
54.
Moreno-Villanueva M, Wong M, Lu T, Zhang Y, Wu H. Interplay of space radiation and microgravity in
DNA damage and DNA damage response. NPJ Microgravity. 2017; 3:14. https://doi.org/10.1038/
s41526-017-0019-7 PMID: 28649636.
55.
Chaudhry MA. Real-time PCR analysis of micro-RNA expression in ionizing radiation-treated cells.
Cancer Biother Radiopharm. 2009; 24(1):49–56. https://doi.org/10.1089/cbr.2008.0513 PMID:
19216629.
56.
Simone NL, Soule BP, Ly D, Saleh AD, Savage JE, Degraff W, et al. Ionizing radiation-induced oxida-
tive stress alters miRNA expression. PLoS One. 2009; 4(7):e6377. https://doi.org/10.1371/journal.
pone.0006377 PMID: 19633716.
Systemic response to spaceflight
PLOS ONE | https://doi.org/10.1371/journal.pone.0199621
July 25, 2018
19 / 19
