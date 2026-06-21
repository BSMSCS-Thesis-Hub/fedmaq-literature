1

## Non-IID data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions

Daniel M. Jimenez G. , David Solans , Mikko A. Heikkil¨ a , Andrea Vitaletti , Nicolas Kourtellis Aris Anagnostopoulos , Ioannis Chatzigiannakis , Senior Member, IEEE

Abstract -Recent advances in machine learning have highlighted Federated Learning (FL) as a promising approach that enables multiple distributed users (so-called clients) to collectively train ML models without sharing their private data. While this privacy-preserving method shows potential, it struggles when data across clients is not independent and identically distributed (non-IID) data. The latter remains an unsolved challenge that can result in poorer model performance and slower training times. Despite the significance of non-IID data in FL, there is a lack of consensus among researchers about its classification and quantification. This technical survey aims to fill that gap by providing a detailed taxonomy for non-IID data, partition protocols, and metrics to quantify data heterogeneity. Additionally, we describe popular solutions to address non-IID data and standardized frameworks employed in FL with heterogeneous data. Based on our state-of-the-art survey, we present key lessons learned and suggest promising future research directions.

Index Terms -federated learning, data heterogeneity, non-IID data, distributed learning, partition protocols, non-IID metrics

## I. INTRODUCTION

Federated Learning (FL) [1], [2] represents a paradigm shift in the development of machine learning (ML) models, particularly in response to the growing need for data privacy and the distributed nature of modern datasets. Traditionally, ML frameworks have been designed on the assumption that data is centralized, residing in a single, unified repository. This centralized approach allows for directly aggregating data from diverse sources, facilitating the training of models on large, comprehensive datasets. However, such a methodology introduces several challenges, mainly when dealing with sensitive or proprietary data. Data centralization can lead to privacy concerns, increased security risks, and logistical difficulties associated with transferring and storing vast amounts of information.

As data becomes increasingly decentralized (a.k.a. federated), residing on edge devices such as smartphones or within isolated institutional silos (so-called clients), limitations of traditional ML approaches such as privacy and efficiency have become more pronounced. Data privacy regulations, such as the General Data Protection Regulation (GDPR) [3] and the Health Insurance Portability and Accountability Act (HIPAA) [4], impose substantial requirements on collecting and processing personal data, making the centralized aggregation of such data impractical or legally infeasible.

FL addresses these concerns by enabling collaborative model training across decentralized data sources without requiring raw data to leave its local environment. In FL, local models are trained independently on each client, and only the learned model updates (e.g., model gradients or weights) are shared with a central server. The central server then aggregates these updates, typically through weighted averaging, to create a global model incorporating knowledge from all participating nodes. Crucially, no raw data is exchanged, which helps preserve privacy and mitigate the risks associated with data breaches or leakage during transmission.

This decentralized approach is particularly suited to applications where data heterogeneity, privacy, and security are paramount, such as healthcare [5], finance [6], and mobile computing [7]. FL thus provides a robust solution to the challenges posed by the distributed nature of modern data while maintaining the utility of machine learning models at scale.

Throughout this paper, we highlight key findings from our survey using light blue background text to facilitate quick identification of significant insights.

## A. Motivation

In FL research, there is a notable lack of publicly available datasets that are inherently federated and effectively capture the diverse and non-centralized data distributions typical of real-world environments. Consequently, a widely adopted approach in the literature involves simulating the federated setting by partitioning existing centralized datasets among several synthetic clients. This explicit partitioning allows for emulating a decentralized learning environment across multiple clients while retaining the benefits of a controlled experimental scenario.

Typically, for this synthetic data partitioning, centralized datasets such as CIFAR-10 [8], MNIST [9] or FMNIST [10] are divided into non-overlapping datasets, which are then assigned to individual clients. Usually, various types of data skewness, such as label imbalance, feature distribution divergence, and quantity skew, can be systematically generated through specific partition protocols. We use the term partition protocol (a.k.a partition method) to refer to the systematic partitioning functions used to divide a dataset into smaller subsets, leading to a federated data environment. These partition protocols are crucial in simulating meaningful experimental scenarios for FL. Depending on the protocol, the resulting distributions can either follow an IID (independent and identically distributed) scheme, where each client receives data that mirrors the overall distribution of the dataset, or a non-IID scheme (a.k.a. non-IIDness), where data is distributed in a skewed or heterogeneous manner across clients.

TABLE I: Comparison of previous surveys and reviews for non-IIDness in FL ( /check-circle : Included, /minus-circle : Partially included, /times-circle : Not included)

| Survey or Review   |   Publication Year | non-IID data Taxonomy   | Partition Protocols   | non-IID Metrics   | Modality Skew   | Solutions to non-IIDness   | Standardized Frameworks   |
|--------------------|--------------------|-------------------------|-----------------------|-------------------|-----------------|----------------------------|---------------------------|
| [11]               |               2024 | /times-circle           | /times-circle         | /times-circle     | /times-circle   | /check-circle              | /times-circle             |
| [12]               |               2024 | /times-circle           | /times-circle         | /times-circle     | /minus-circle   | /check-circle              | /times-circle             |
| [13]               |               2024 | /minus-circle           | /times-circle         | /times-circle     | /times-circle   | /check-circle              | /times-circle             |
| [14]               |               2024 | /check-circle           | /times-circle         | /times-circle     | /times-circle   | /check-circle              | /times-circle             |
| [15]               |               2022 | /minus-circle           | /times-circle         | /times-circle     | /times-circle   | /minus-circle              | /times-circle             |
| [16]               |               2022 | /minus-circle           | /minus-circle         | /times-circle     | /times-circle   | /check-circle              | /times-circle             |
| [17]               |               2021 | /check-circle           | /minus-circle         | /times-circle     | /times-circle   | /check-circle              | /minus-circle             |
| Ours               |               2024 | /check-circle           | /check-circle         | /check-circle     | /check-circle   | /check-circle              | /check-circle             |

The choice of a partition protocol significantly impacts the performance and convergence of FL algorithms, as non-IID data often presents additional challenges for model generalization and parameter aggregation compared to an IID data setting. Thus, designing and selecting appropriate partition protocols is critical to evaluating the robustness and effectiveness of FL methods.

## B. Previous surveys on non-IID data in FL

However, despite the critical importance of establishing a solid understanding of the existing types of data skewness in FL and the necessity of using standardized approaches to simulate realistic data partitions, few studies have thoroughly addressed the need to create a meaningful and comprehensive categorization of existing data partitioning techniques in FL. While we have identified a few surveys that provide foundational insights into the challenges of data heterogeneity, our work advances beyond these contributions by offering an extended taxonomy alongside a more detailed overview of existing protocols and metrics.

We present previous surveys and reviews in Table I, which reveals a clear evolution in the coverage of non-IID data aspects from 2021 to 2024. Earlier surveys from 2021-2022 demonstrate partial or limited coverage across most categories, although they generally include solutions to non-IIDness. Recent surveys from 2024 still maintain significant gaps, particularly in the inclusion of partition protocols, non-IID metrics, modality skew, and standardized frameworks (see Table II). Our work includes complete coverage of all the aspects related to non-IID data and represents a significant advancement towards understanding, quantifying, and tackling non-IIDness in FL.

## C. Contribution

This work is the first technical survey dedicated to organizing and synthesizing the existing knowledge regarding distri- bution skewness in FL, providing a comprehensive taxonomy and detailed categorization of data heterogeneity types. We specifically contribute by:

- 1) Introducing a taxonomy that organizes and synthesizes the diverse forms of non-IIDness.
- 2) Reviewing state-of-the-art partition protocols for simulating non-IID scenarios in FL.
- 3) Methodically analyzing non-IID metrics for quantifying data heterogeneity, fostering standardized evaluations.
- 4) Offering good practices to ensure experimental consistency, fair benchmarking, and realistic deployment testing.

Our work bridges gaps in existing surveys, serving as a critical reference for researchers and practitioners in FL.

## D. Paper structure

The rest of this survey is organized as follows: Section II provides an overview of the basic notions of FL. Section III describes the approach utilized to retrieve the most relevant papers this survey. Section IV categorizes the various types of data skewness considered in the state of the art. Section V overviews the landscape of partition protocols and heterogeneity metrics utilized in previous research. Section VI reports existing state-of-the-art solutions to tackle non-IIDness in FL. Section VII collects information about existing standardized frameworks for FL and their capacities to deal with heterogeneous data. Section VIII compiles a collection of lessons learned and a list of good practices based on the knowledge acquired during the execution of this work. Section IX summarizes the future work vision from the various research efforts analyzed. Finally, Section X synthesizes the key findings from our survey and overviews the potential directions for future work in the field. Fig. 1 illustrates the organizational structure of our survey.

## II. FEDERATED LEARNING BASICS

FL is an innovative ML paradigm that enables training models on distributed datasets across multiple clients holding local data samples (a.k.a examples, individuals, data points) without exchanging them. This approach contrasts traditional centralized ML techniques, where all the data get aggregated in one location for model training.

Title: Non-IID data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions

Fig. 1: Outline of our survey

<!-- image -->

FL minimizes a global objective function l ( w ) , represented as a weighted average across the private datasets of all participating clients. In this framework, we consider a scenario with K total clients, where each client k possesses its private dataset D k . Equation Eq. 1 defines the function minimized,

$$
\begin{aligned}
\min _ { w } l ( w ) \colon = h ( L _ { k } ( w ) ) & & ( 1 ) & & 1 )
\end{aligned}
$$

where w represents the parameters (a.k.a. weights) of the global model trained, and h is function to aggregate the clients parameters. L k ( w ) is a local objective function such as crossentropy loss for a supervised classification task [11].

The core idea behind FL is to train the global model w by aggregating the local model updates (e.g., model gradients or weights) from the participating clients. The latter occurs through a process called Federated Averaging ( FedAvg ) [1], [18], where each device trains its local model on its data, and the central server aggregates the model updates from the devices to update the global model. In FedAvg the aggregation function h ( L k ( w )) = ∑ K k =1 n k n L k ( w ) , where w represents the parameters of the global model trained, n k represents the size of D k , and n is the total number of samples held by all participating clients. The mentioned iterative process of local training and global model aggregation continues for a predefined number of communication rounds ( T ), allowing the model to be trained on a diverse range of data sources while preserving the privacy of the individual data contributors, as the raw data never leaves the local devices. Alternative aggregation functions to FedAvg in FL are explained in Section VI.

## A. Federated Learning types

FL gets categorized into different types based on the participating entities. These participants can range from individual mobile devices and IoT sensors (cross-devices) to large organizations and institutions (cross-silos). The following paragraphs explain each type in more detail [19].

- 1) Cross-silos: It involves collaboration between a limited number of organizations or institutions, typically with relatively powerful computational resources. In this setting, each client (or 'silo') has a substantial amount of data and robust infrastructure. Examples include hospitals sharing medical data for research or banks collaborating on fraud detection models while maintaining client privacy [20].
- 2) Cross-devices: Cross-device FL operates on a much larger scale, involving numerous edge devices such as smartphones, IoT devices, or wearables. This type of FL deals with many participants with limited computational resources and smaller, often intermittently available datasets. Cross-device FL must handle challenges like unreliable network connections, device heterogeneity, and frequent client unavailability. Its use is commonly in mobile keyboard prediction, voice assistants, and other consumer-facing applications [20], [21].
- 3) Hierarchical FL: Hierarchical FL introduces a multilevel structure to the FL process, typically involving cross-silo and cross-device elements. In this approach, devices or lowerlevel participants aggregate their models within local clusters or organizations before further aggregation occurs at higher levels. This structure can improve efficiency in large-scale systems, reduce communication overhead, and allow for more

flexible data-sharing policies. Hierarchical FL is beneficial in scenarios involving multi-national corporations, healthcare networks, or large-scale IoT deployments [22].

Regarding the data disposition, FL can be categorized primarily into horizontal and vertical partitions. The data disposition pattern influences aspects such as model architecture, aggregation methods, and privacy-preserving techniques employed in the FL system. The following paragraphs define each type thoroughly [23].

- 4) Horizontal FL: Horizontal FL, also known as samplebased FL, is applicable when different participants have datasets with the same feature space but different sample sets. For instance, two regional banks might have the same types of customer data (features or attributes) but for different sets of customers (samples) [24].
- 5) Vertical FL: Vertical FL, or attribute-based FL, is used when participants have datasets with identical samples but different feature spaces. This scenario often arises in multiparty business collaborations. For example, a bank and an ecommerce company might have different data types (attributes or features) for the same set of customers [25].
- 6) Peer-to-peer FL: Also known as decentralized FL, removes the central server, using peer-to-peer communication for model aggregation via epidemic or diffusion processes [26]. This approach improves scalability, fault tolerance, and privacy, making it ideal for edge networks or blockchain-based systems. Unlike hierarchical FL, it relies on flexible, pairwise, or group-based updates.

## B. Non-IID data in FL

In centralized learning, non-IID data typically refers to variations within a single dataset that violate the assumption of IID [17], [27]. The latter might include class imbalances, temporal shifts in data distribution, or biases in data collection. Although challenging, researchers often address such issues using data augmentation [28], resampling [29], or careful stratification during train-test splits [30]. In contrast, nonIID data in FL presents a more complex scenario. In FL, not only each participant's local dataset can be non-IID, but the data distribution across participants can also be highly heterogeneous. It leads to a two-level non-IID problem: intraclient non-IID (within each participant's dataset) and interclient non-IID (across different participants). The decentralized nature of FL, combined with privacy constraints that limit direct data sharing, makes traditional centralized approaches to handling non-IID data insufficient [31].

## C. Downstream effects of non-IID data in FL

Non-IID data in FL can significantly affect the learning process's performance, efficiency, and reliability. These consequences include but are not limited to:

- Model bias: The model may become biased towards the data distribution of participants with more prominent or influential datasets [32].
- Slower convergence: Non-IID data can increase training time as the global model struggles to reconcile divergent local updates [33].
- Reduced model accuracy: The global model may underperform on specific subsets of underrepresented data in the federated dataset [34].
- Instability in training: Non-IID data can cause fluctuations in model performance across training rounds, making it difficult to achieve consistent improvements [35].
- Challenges in participant selection: Non-IID data can complicate choosing representative participants for each training round [36].
- Communication inefficiency: More communication rounds may be required to reach convergence, increasing network overhead [37].
- Privacy risks: In some cases, non-IID data can make inferring information about individual participants' datasets easier through model updates [38].
- Susceptibility to attacks: Non-IID data can aggravate vulnerabilities to cybersecurity threats unique to FL, such as poisoning attacks, where adversarial participants introduce malicious updates to compromise the global model [39].

FL systems must employ tailored strategies like dynamic aggregation approaches, individualized modeling techniques, and resilient federated optimization methods to mitigate these issues [40]-[42]. These specialized solutions facilitate effective learning from diverse data distributions while preserving data privacy and maintaining computational efficiency.

Fig. 2: Research interest in general FL vs. non-IIDness in FL

<!-- image -->

The resulting challenges and complexities make non-IID data in FL relevant for researchers. In this context, Fig.2 depicts how research focusing specifically on non-IID challenges in FL has notably increased since 2020, also peaking in 2023, highlighting the growing attention this issue is receiving. It is essential to emphasize that the apparent decrease in 2024 is due to the timing of our data collection, as papers were gathered during the third quarter, leading to incomplete data for that year (see Section III)). Thus, although general FL research continues to expand more quickly, the consistent rise in studies addressing non-IID data shows that it is emerging as a critical topic within the FL research community [43].

## III. LITERATURE IDENTIFICATION AND META-ANALYSIS

In this section, we outline the methodology used to identify and select the relevant papers and articles that form the basis of our survey. Additionally, we present a detailed metaanalysis of these papers, highlighting compelling information and insights.

## A. Methodology for identifying relevant literature

We employed a methodology for identifying relevant literature based on the PRISMA methodology [44] to conduct a thorough and structured analysis of the current literature on non-IID-ness in FL. We improved the latter methodology by considering top-tier published papers and research from highlevel universities. The approach involved gathering, filtering, and assessing relevant academic papers and resources from various scholarly databases. The process began with an identification phase, as illustrated in Fig. 3. We utilized six reputable academic repositories: Google Scholar [45], IEEE Xplore [46], PubMed [47], Scopus [48], and Web Of Science [49]. We chose these platforms for their comprehensive coverage across multiple academic fields and their extensive collection of research publications. Our search strategy involved 83 queries across ten categories related to non-IID-ness in FL, including general non-IID-ness, label skew, attributes skew, quantity skew, distribution shift, non-IID metrics, performance metrics, partitioning protocols, datasets, and frameworks. We input each of the 83 queries into each academic repository to retrieve relevant papers. After eliminating duplicate documents, we compiled a collection of 5489 unique papers for further analysis.

Fig. 3: PRISMA flow for gathering relevant references

<!-- image -->

We screened each article after the initial collection to identify potentially valuable papers. Our screening process consisted of the following steps for each paper retrieved:

- 1) We examined the title and available keywords of every paper.
- 2) If these elements were relevant (for instance, if they addressed non-IID-ness, discussed heterogeneity, mentioned label skew, or compared centralized and federated approaches), we marked the paper as 'possibly useful' for further review.

The latter allowed us to filter the large pool of papers, focusing on those most likely to contribute to our study of non-IID-ness in FL.

In the eligibility phase, we refined the selection of references to focus on the most relevant and impactful works related to non-IID-ness in FL by reviewing the titles and abstracts and identifying the conference ranking or journal quartile. As a result, 202 papers were marked as 'useful' if they met either of the following criteria:

- 1) The title and abstract address non-IID-ness.
- 2) The paper was published at a conference ranked A* or A or in a Q1 journal.

Some relevant papers are often not published in conferences or journals. Still, it is worth analyzing whether they have received many citations or come from high-ranking universities. Then, 36 additional papers are marked as 'useful' if it was not published in a journal or conference (i.e., published in Arxiv) and any of the following occurs:

- 1) For papers from 2024, it has been cited (using the normalized-by-year citations count) at least once.
- 2) For papers before 2023 included, it was cited more than four times for other years (using the normalized-by-year citations count).
- 3) The author's universities included in the paper are in the top 100 universities regarding the QS ranking -Engineering and Technology- [50].

After applying the rigorous eligibility process described, we included 235 papers to serve as the foundation for our survey.

## B. Meta-analysis of relevant literature

We comprehensively examined the 235 papers selected through our process to provide a general glance at the current research on non-IID-ness in FL. Through this analysis, we seek to contextualize individual findings within the broader landscape of non-IID FL literature and provide a foundation for understanding the collective progress and challenges in this rapidly evolving domain.

Fig. 4: Rankings and quartiles distribution comparison

<!-- image -->

Fig. 4 presents a comparative analysis of the publication quality between the papers included in our final study and those initially retrieved during the screening phase. It demonstrates a clear shift towards higher-quality publications in our selected ones. Specifically, the included papers predominantly originate from top-tier conferences (A* and A rankings) and high-impact journals (Q1 quartile). This notable difference in quality distribution underscores the efficacy of our rigorous selection criteria and eligibility process.

Fig. 5: Geographic distribution of collected papers by first author affiliation country. Map values indicate each country's relative prevalence inside the collected database.

<!-- image -->

The world map of Fig. 5 depicts the geographic distribution of collected papers based on the first author's affiliation. Highlighted in black, China emerges as the dominant contributor, indicating its leading role in this research field. North America (the United States and Canada) and Australia are the second most significant contributors. Other Asian countries (i.e., South Korea, Singapore, and Japan, among others) and Europe (Germany, Italy, Spain, etc.) are represented in lighter shades with less research participation. Those countries in white represent the lack of research contribution. This map reveals a precise concentration of research output in specific regions, potentially reflecting global differences in research focus, funding, or technological infrastructure.

Fig. 6: Percentage of research done in academic and industrial environments

<!-- image -->

Fig. 6 illustrates the distribution of research conducted in academic versus industrial environments. It reveals that academia dominates the research landscape, accounting for 65.1% of the studies. Interestingly, there's an overlap of 30.6% between academic and industrial research, suggesting a considerable amount of collaborative work or researchers with dual affiliations. A small segment of 4.3% represents research conducted solely in industry. It underscores the importance of academic institutions in driving research in non-IIDness while highlighting the industry's notable role, mainly through collaborations with universities.

## IV. TAXONOMY OF DATA HETEROGENEITY IN FL

To define different data skews, we assume in this section that all the relevant densities exist for simplicity. We can generally replace densities, e.g., with cumulative density functions below, as long as all the relevant conditional distributions are well-defined. For general terminology, we use skew to refer to various differences between clients' local data sets, and heterogeneity as a more general category that might or might not imply differences between clients.

Writing D i ∼ D i for the i th clients' local data, where D i represents the underlying data distribution from which the i th client's local data D i is sampled, and denoting the probability density function (pdf) of D i by f ( i ) , data skew means that

̸

$$
f ^ { ( i ) } \neq f ^ { ( j ) } \text { for some } i , j \in \{ 1 , \dots , K \} .
$$

In other words, this means that the local data distributions differ somehow between at least one pair of clients. Data skew is a challenge in FL because it can lead, e.g., to drift or bias in the local models, complicating the aggregation process and hindering the performance and generalization of the global model across clients. 1

In the supervised learning setting with D i = ( y i , x i ) , where y i are the labels (target outputs or classes assigned to samples) and x i the attributes (features or properties describing samples), we can further divide general data skew into label and attribute skews as in the following:

- 1) Label skew. We have marginal or conditional label skew , respectively, if

̸

$$
f _ { Y } ^ { ( i ) } \neq f _ { Y } ^ { ( j ) } \text { or }
$$

̸

$$
f _ { Y | X } ^ { ( i ) } \neq f _ { Y | X } ^ { ( j ) } \text { for some clients } i , j
$$

̸

i.e., the marginal f ( i ) X or conditional f ( i ) Y | X distributions of the labels differ between some pairs of clients. In general, with heterogeneous data we might find either marginal or conditional skew or both. However, if there is no attribute skew (see below), then marginal label skew implies conditional label skew and vice versa. More formally, assuming f ( i ) X = f ( j ) X and f ( i ) X | Y = f ( j ) X | Y (no attribute skew) implies f ( i ) Y = f ( j ) Y ⇔ f ( i ) Y | X = f ( j ) Y | X . The proof follows immediately from Bayes' Theorem:

̸

$$
\begin{aligned}
f _ { Y | X } ^ { ( i ) } = \frac { f _ { X | Y } ^ { ( i ) } f _ { Y } ^ { ( i ) } } { f _ { X } ^ { ( i ) } } & & ( 5 )
\end{aligned}
$$

which readily implies the claim when comparing to client j and canceling out the common factors.

As illustrated by Fig. 7, we further divide label skew into four different subtypes based on the analyzed papers. (1) Distribution-based skew refers to a situation where clients share classes but the prevalence of each class is ruled by some defined distribution (e.g., client A might have 80% cats and 20% dogs, while client B has 30% cats and 70% dogs) [51]-[60]. (2) Overlapping class allocation describes scenarios where multiple clients share some common classes but each client may also have some unique classes (e.g., client A has classes 1,2,3 while client B has classes 2,3,4) [61]-[64]. (3) Non-overlapping class allocation represents a more extreme case where different clients have completely disjoint sets of classes with no overlap between them (e.g., Client A has classes 1,2 while Client B has classes 3,4) [65]-[68]. Finally, (4) Noisy labels refers to cases where some data points in clients' datasets are incorrectly labeled, adding noise to the FL process [69]-[71].

1 Note that data skew is a theoretical concept that pertains to the local data-generating processes. It is not directly observable with finite empirical datasets, and testing for it involves inherent uncertainty. On a practical level, existing research uses various methods to simulate data skew empirically, which we also cover in the following.

Fig. 7: Taxonomy for non-IIDness in FL. The figure categorizes different types of data skews and heterogeneities. Percentages indicate the proportion of each type across global, cross-silo-related, and cross-device-related papers.

<!-- image -->

- 2) Attribute skew. Similarly as with the labels, we can look at the marginal and conditional distributions of the attributes and have marginal attribute skew or conditional attribute skew , respectively, when

̸

$$
\begin{aligned}
f _ { X } ^ { ( i ) } \neq f _ { X } ^ { ( j ) } \text { or } & & ( 6 ) & \text { as } \text { te}
\end{aligned}
$$

̸

$$
f _ { X | Y } ^ { ( i ) } \neq f _ { X | Y } ^ { ( j ) } \text { for some clients } i , j . \quad ( 7 ) \quad \text {total} \ \text {client}
$$

Mirroring the situation with the labels above, if there is no label skew, then marginal attribute skew implies conditional attribute skew and vice versa.

As depicted in Fig. 7, we divide attribute skew into four subtypes based on our analysis of the existing research. (1) Distribution-based skew refers to situations where clients share the same features but have different probability distributions over these features (e.g., different distributions of pixel values in images across clients) [72]-[77]. (2) Partial attribute selection describes scenarios where clients have incomplete or partially observed features for their samples [78], [79]. Noisy attributes represents cases where the feature values in some clients' datasets contain noise or corrupted values, affecting the data quality [80], [81]. (4) Vertical skew , also known as vertical FL, refers to situations where different clients have different subsets of features for the same samples (e.g., Client A has features 1-10 while Client B has features 11-20 for the same set of instances) [82][86].

Unlike most previous surveys, we include the notion of Modality skew under the taxonomy of non-IIDness. It is a specific type of data skew referring to a setting where different clients have data from varying input modalities, such as text, images, audio, tabular, or sensor data, as opposed to all clients having data of the same type. This occurs when clients collect or generate data in distinct formats, leading to heterogeneity in the input data types across the network. Modality skew is a subtype of the attribute and label skews whose interdependence is detailed in Fig. 8.

As represented in Fig. 7, we distinguish between two subtypes of modality skew based on our analysis. (1) Complete multimodality skew refers to a scenario where all clients have the same types of multimodal data (e.g., all clients have both image and text data), but there might be differences in the distribution of these modalities across clients [87], [88]. For example, one client might have a different distribution of image-text pairs compared to another client while still maintaining all modalities. In contrast, (2) partial multimodality skew represents a situation where clients have varying subsets of the possible modalities (e.g., some clients might have both image and text data, while others might only have image data) [89], [90].

Besides the main forms of data skew discussed above, which are based on differences in the local data distributions between clients, we further identified quantity skew, spatiotemporal heterogeneity, and participation skew as contributing to the general data heterogeneity in FL. However, as detailed next, these forms of heterogeneity differ in essential ways from the forms of data skew defined above.

- 1) Quantity skew. Another important factor of data heterogeneity in FL is the possibly wildly different amount of data each client holds. We call this type of data heterogeneity quantity skew . Unlike data skew, the definition of which is based on the local data generating processes, quantity skew is a purely empirical quantity. It is orthogonal to data skew, and the actual effect depends on whether there is data skew present or not.

As shown in Fig.. 7, quantity skew only contains the subtype called sample-based skew , which refers to differences in the number of samples among the clients [91][99]. Moreover, our analysis of the literature identified a related concept known as data sparsity . This term describes a specific form of sample-based quantity skew, where certain clients systematically lack data for specific features, classes, or patterns [100]-[104]. For instance, in the context of medical records, data sparsity might manifest as some clients frequently having missing blood test values or incomplete patient histories. As such, data sparsity can be interpreted as a specific realization of data skew.

- 2) Spatiotemporal heterogeneity. This type of heterogeneity arises when data across clients differs in either or both spatial or temporal dimensions. As presented in Fig. 7, following here the common trend in the analyzed papers, by (1) temporal heterogeneity we simply mean that the local data on a given client form a time series, i.e., the local samples on a given client have a temporal dimension instead of being IID samples [105]-[107]. This is orthogonal to having data skew in the above sense, as the stochastic process generating the local data might be the same (implying no data skew) or different between clients. When the learning algorithm or the clients only have access to data corresponding to a single time-step at a time, this can lead to the federated continual learning problem [108], [109].

In turn, and again following the usage in the analyzed papers, by (2) spatial skew we refer to differences in client data distributions (i.e., having some form of data

- skew) caused by differences in the physical location of the clients [110]-[112].
- 3) Participation skew. We refer to imbalances in client participation during the training or testing/production phases in FL as participation skew . This skew arises from two key factors: client selection, i.e., from how and when the clients are included in the training, and client dropouts, where some otherwise included clients may intermittently fail to participate in training due to network issues, device unavailability, or other constraints. As a result of participation skew, not all clients contribute equally or consistently to model training. In the presence of data skew, such imbalances can give rise to biases in the global model that reflect the over-representation of certain clients while others are underutilized or excluded. This can lead to various issues, such as different forms of biases (for example, when some group of clients has systematically worse utility) and problems with out-ofdistribution (OOD) generalization (e.g., when there is data skew between training and testing/production set of clients).

Regarding Fig. 7, participation skew occurs in two types that we define according to the analysis of the papers. (1) Party selection and subsampling refers to the strategic or random selection of a subset of clients to participate in each training round, which can be influenced by factors such as computational resources or communication constraints (e.g., only selecting 10 out of 100 available clients per round) [113], [114]. (2) Client dropout describes scenarios where clients unexpectedly become unavailable or disconnect during the training process, which can happen due to network issues, device limitations, or other technical problems (e.g., a client might lose connection mid-training or run out of battery) [115], [116].

## A. Prevalence of non-IIDness types in FL

As shown in Fig. 7, label skew is the most prevalent type of heterogeneity studied in the existing literature with a higher prevalence in cross-device than in cross-silo settings. Quantity skew is the second most studied type, with the same prevalence in cross-silo and cross-device studies. Despite its importance, spatiotemporal and especially temporal heterogeneity is not broadly treated in the papers analyzed. We additionally note that modality skew is an understudied topic, with only 2% of the papers including it.

Looking at the subtypes within each category reveals interesting patterns. For instance, within label skew, distributionbased skew (48-55%) and non-overlapping class allocation (36-52%) are much more common than overlapping class allocation (9-19%) and noisy labels (2-3%). In the attribute skew category, vertical skew (38-50%) dominates other subtypes. The latter occurs because vertical skew relates directly to vertical FL (see Section II) which is relatively more prevalent in real-world datasets than the other subtypes. For participation skew, party selection and subsampling as well as client dropout show significant variation across contexts (0-67%, 0-38%, respectively). This is mainly explained by the fact that neither form of participation skew is typically relevant for cross-silo studies, as in most cases all clients are included on every training and testing round, and the clients are assumed to have good connectivity.

These differences in prevalence suggest that certain types of non-IIDness have been studied much more than others. For example, label skew is significantly more common than attribute skew in the papers we have analyzed. This could guide researchers and practitioners in prioritizing which challenges to address in research as well as in designing practical FL systems.

## B. Interdependence of skew and heterogeneity types

In previous sections of this manuscript, we provide an individual overview of the various types of data heterogeneity that can occur in FL. In practice, the various types of data heterogeneity in FL do not operate in isolation. Instead, they can exhibit complex interdependencies, where one form of heterogeneity can directly cause or modify the impact of another.

Fig. 8: Interdependencies between skew types in FL

<!-- image -->

As illustrated in Fig.8, data skew encompasses both label skew and attribute skew, which can occur independently or simultaneously, such as bank A having mostly legitimate transactions (label skew) and bank B having high-value transfers (attribute skew). Additionally, modality skew represents a subset of attribute skew and may coincide with label skew, as occurs when Hospital A has X-rays of pneumonia cases while Hospital B has computed tomography scans of healthy patients.

On the other hand, other types of heterogeneity may influence, induce, or worsen the problems generated by these core skew types. As an example, running FL training with clients physically located in different places (spatial heterogeneity)

can cause data skew between clients, which can further combine with participation skew (e.g., when only recharging mobile devices can take part in training and charging is more common during clients' night time) to induce notable biases in the global model. Another related concept is temporal heterogeneity, which can make learning harder also in the FL setting even without any data skew.

Furthermore, any form of data heterogeneity possible in the centralized data settings can also appear in the FL context. As an example, there might be differences in the distributions between training and testing/production data sets, which can then lead to OOD generalization issues with centralized data. Similarly, differences between training and test/production data sets might appear in FL, either systematically inside each client's data, in which case there need not be any betweenclients data skew or with data skew. Such combinations can arise in practice, e.g., due to the temporal difference between training a model and using it in production, even when the sets of training and production clients would be essentially the same.

Tackling these interrelated skews demands a comprehensive approach that accounts for how each type contributes to data heterogeneity and affects the learning performance in federated models.

## V. METHODS AND METRICS FOR DATA HETEROGENEITY IN FL

This section provides state-of-the-art standardized partition protocols to split centralized data into federated datasets and metrics to quantify non-IIDness among clients.

## A. Standardized dataset's partition protocols

Protocols for dataset partitioning in FL are crucial for advancing research and practical applications. The current lack of consensus on the sufficient conditions and scenarios for testing solutions to non-IID data has led to an increase of partitioning protocols and experimental setups. This inconsistency complicates the replication of results, validation of findings, and comparison of different approaches. Establishing agreedupon methods for creating, quantifying, and evaluating nonIID partitions is essential for enabling fair comparisons, enhancing reproducibility, and providing a common framework for describing various non-IID scenarios.

Leveraged on the exhaustive analysis of the selected papers from the survey, we created a taxonomy for the partition protocols included in the recent literature. Fig. 9 depicts the classification encountered for the methods used in the non-IID research field.

In the following paragraphs, we define each category of the taxonomy presented in Fig. 9, together with the explanation of the detailed methods to partition centralized data into federated data.

1) Label skew : It refers to partitioning the centralized dataset considering the uneven distribution of class labels across different clients in an FL setting as exposed in Section IV. It is divided into three subcategories: Distributionbased, Class allocation, and Noise-based.

Fig. 9: Taxonomy for partition protocols in Federated Learning

<!-- image -->

a) Distribution-based : This approach allocates data to clients based on a specified probability distribution of labels to create imbalanced label distributions across clients.

The Dirichlet partition protocol is the most employed in research, with 27% of the papers implementing it [75], [117][124]. It uses the Dirichlet distribution (DD) [125] to allocate samples across clients, simulating non-IID scenarios. A concentration parameter α controls the degree of data heterogeneity. Lower values create more skewed distributions, while higher values approach IID conditions. This method permits generating a range of realistic non-IID datasets for evaluating FL algorithms under varying degrees of data imbalance. The DD generalizes the Beta distribution, which generalizes the Uniform distribution. As a result, it leads to a skewed data split [126].

Another method in this category is the Generator of NonIID datasets [127]. It receives as inputs the minimum and maximum number of data points that can be assigned to any partition ( D min, D max), the number of labels (or classes) to be sampled for each partition ( L num) and the number of partitions ( P ). Then, it creates a set of data partitions Dataset = ( D 1 , D 2 , . . . , D P ) . For each partition i , it randomly samples L num labels from the set of unique categories. It then selects a random number D num within the range [ D min , D max ] . A set of weights is sampled from the interval (0 , 1) , normalized by dividing each by the sum of all weights. Each class's total number of data points is calculated by multiplying D num by these normalized weights. Finally, data is sampled according to this computed number of data points and the chosen classes, generating the non-IID datasets for each partition.

- b) Class allocation : This method assigns entire classes or datasets of classes to different clients. It has two variations:
2. i . Overlapping classes: In this variation, some classes are shared among multiple clients, while others may be unique to specific clients.

The Percentage-of-non-IID-ness method [128]-[132] is the one employed the most in this category, reaching 7% of the papers. It controls the skewness by adjusting the fraction of data that is non-IID. For example, 20% non-IID indicates that 20% of the dataset is partitioned based on labels, while the remaining 80% is partitioned uniformly at random. While varying the percentage, different levels of skewness in the dataset distribution can be achieved, allowing for controlled experiments with different degrees of heterogeneity.

Another method is the Dominance class Ratio [40], [133][136], which has a statistical parameter σ ∈ [0 , 1] to control the ratio of the dominant class within each client. The latter indicates that one class primarily dominates within each client, while the remaining classes are uniformly distributed across the clients. Adjusting σ permits simulating different degrees of class dominance and distribution imbalances.

One alternative method falling in this category is the local long-tail (LLT) partition, which simulates Non-IID datasets with a long-tail distribution [137], [138]. This method divides a global dataset D with C classes among K clients (where K = C ). The parameter α l controls the degree of non-IIDness. For each class c ∈ C , the samples get partitioned into K parts, with one part containing α l N ∗ ,c samples assigned to a specific client k and the remaining K -1 parts containing 1 -α l K -1 N ∗ ,y samples distributed to other clients. The latter ensures each client has one dominant class and C -1 tail classes. The LLT partition reflects real-world scenarios, like smartphone users favoring certain photo styles, but may not fully capture the distribution characteristics seen in real-world FL tasks. Despite this, the LLT method mirrors imbalanced learning scenarios, serving as a basis for testing data resampling effects in FL.

ii . Non-overlapping classes: Regarding this classification, each client is assigned distinct classes, with no overlap between clients.

The most used partition protocol in this category is the Sharding method [139]-[144], employed by 20% of the papers. It is a technique used to create non-IID datasets by sorting them based on class labels and then dividing them into smaller datasets (a.k.a shards) of equal size. These shards are distributed among clients, each receiving a fixed number. Since each shard contains data from only a few classes, clients have data representing only a small subset of the overall class distribution. This method results in a highly skewed, non-IID distribution, as clients predominantly possess samples from just a few classes, and these do not overlap among the clients. Notice that the case when each client holds only one class is known as Pathological partition [140], [145], [146].

The Archetypes method [147], [148] divides the dataset into distinct groups, or archetypes, where each group contains data corresponding to a subset of class labels. Clients are assigned to one or more archetypes, meaning that each client only has access to a specific subset of the overall class distribution. For instance, one archetype may consist of data with labels { 0, 1, 2, 3, 4 } , while another includes labels { 5, 6, 7, 8, 9 } . This method results in a structured Non-IID distribution where clients' data is restricted to a limited range of classes.

- c) Noise-based : This technique introduces label noise to simulate label skew, including Symmetric and Asymmetric noise.

The symmetric [71], [149], [150], and asymmetric [70] label noise methods introduce noise into the labels to simulate mislabeling. In the symmetric case method, the correct labels of each class are randomly flipped to any of the other categories with a fixed probability, ensuring an even distribution of incorrect labels across all other classes. In contrast, the asymmetric method involves mapping the correct labels to a specific confusion class, which is more likely to be mistaken for the original label. This results in a more structured type of noise, where mislabeling follows a particular pattern based on class similarities.

- 2) Attribute skew : It refers to the simulation of uneven distribution of features or attributes across clients starting from the centralized data. It is divided into three subcategories: Distribution-based, Cluster-based, and Noise-based.
- a) Distribution-based : This approach creates variations in attribute distributions across clients using different parameters for defined probability distributions.

One method falling into this category is the Gaussian distribution affine method [77], which introduces a distribution shift to the training samples at each client by applying an affine transformation. This transformation is randomly generated based on a Gaussian distribution, resulting in variations in the data distribution across different clients. The affine shift alters the data characteristics, such as mean and variance, simulating a more realistic scenario where data collected by different nodes may have other underlying distributions.

Another partition protocol based on attributes is the HistDirichlet [128]. The algorithm starts by characterizing the attributes of the centralized client (i.e., calculating the average of all the features) and categorizing it through a binning process. Next, the distribution of each feature class within each client is determined using the DD with a specified α applied to the binned variable. The latter ensures that the client data is divided into distinct, non-overlapping datasets.

b) Cluster-based : This method uses clustering algorithms to group similar data points and distribute these clusters across clients, creating natural attribute differences.

The clustering method [151] groups the dataset into distinct clusters based on inherent data similarities. These clusters are formed following a non-IID distribution, meaning that the data within each cluster may differ significantly in terms of its features. Once the clustering process is complete, the dataset is divided into datasets corresponding to the clusters, which is helpful for classification or regression tasks. This method ensures that each subgroup reflects a unique data distribution.

- c) Noise-based : Given an initial data federation, it is not genuinely a partition protocol but a method to add attribute differences among clients. It adds noise to the features of data points to create attribute skew, often using Gaussian noise [34], [81], [128], [152], [153] with varying parameters for different clients.
- 3) Quantity skew : It deals with generating uneven distribution of data volume across clients. It is divided into two subcategories: Client distribution-based and Sample distributionbased.
- a) Client distribution-based : This approach allocates varying amounts of data to clients based on specified distri-

butions or rules.

In the Dirichlet for quantity skew method [34], the size of the local datasets |D i | varies across clients, even though the data distribution among them may remain consistent. A DD allocates different data samples to each client. Specifically, it samples q ∼ Dir K ( β ) , where each q j represents the proportion of total data assigned to client C j . The parameter β controls the degree of quantity imbalance. The latter method has one limitation: it is not usable when the data is too small or the number of clients is too big.

The Min-Size-Dirichlet method [128] helps partition data based on the client's data size. It corrects the limitation of the Dirichlet for quantity skew method. The algorithm starts by setting an α value for the DD to generate the participation proportions for each client. A minimum required number of samples is then established for each client, with the minimum proportion size defined as MinSize = MinRequiredSize n , where n represents the total number of samples in the centralized dataset. If any calculated proportions are smaller than MinSize , they are replaced by MinSize . Finally, the proportions are normalized between 0 and 1. The latter ensures the method will converge even when the dataset is small, or the number of clients is significant.

Beutel et.al. [154] implemented inside Flower datasets some power law partition methods, including linear, exponential, and square partitions, to generate splits based on the partition ID. In the linear partitioner, partition sizes increase linearly with the ID, so client 1 gets 1 sample, client 2 gets two samples, and client k gets k samples. The exponential partitioner correlates partition sizes with the exponent of the ID, where client 1 gets ⌊ e ⌋ units, client 2 gets ⌊ e 2 ⌋ , and so on, with leftover data from rounding added to the largest partition. In the square partitioner, partition sizes increase quadratically, so client 1 gets 1 unit, client 2 gets four units, and client k gets k 2 units.

b) Sample distribution-based : This method creates quantity skew by sampling data points for each client according to certain probability distributions or step functions.

In the step partition method [155], [156], each client is assigned a mixture of minor and significant classes, typically comprising more minor classes with relatively smaller datasets and fewer major classes containing larger datasets. The α parameter regulates the ratio between the sizes of the major and minor class datasets. As the value of α increases, the degree of non-IIDness also escalates, leading to a more significant disparity in the data distribution among the clients.

From Fig. 9, we notice that partition protocols simulating and controlling two or three types of data skew simultaneously do not appear in the papers reviewed. This observation reveals a significant gap in current partitioning approaches, as realworld FL scenarios present combinations of different skew types concurrently.

## B. Non-IIDness quantification metrics

The term 'non-IID metric' is used to quantify how much a dataset deviates from being IID, measuring the degree of data heterogeneity. Although research on quantifying nonIIDness in FL is limited, its importance cannot be overstated for several relevant reasons. First, it provides a clear understanding of the statistical diversity among participating clients, directly affecting model convergence, stability, and overall performance [157]. Accurate measurements for measuring non-IIDness help researchers and practitioners anticipate challenges in training, allowing for the design of tailored algorithms to address the specific level of heterogeneity present. These metrics also enable standardized evaluations and benchmarking of the robustness of FL methods across different datasets and non-IID scenarios [158]. Furthermore, they guide the development of standardized testbeds that resemble realworld FL scenarios more accurately to produce robust and generalizable FL solutions.

Based on the comprehensive review of papers selected in our survey, we developed a classification for non-IID metrics in recent literature. Fig. 10 illustrates our taxonomy of measures used to quantify non-IID characteristics in FL research.

Fig. 10: Taxonomy for non-IID metrics in Federated Learning

<!-- image -->

In the subsequent subsections, we provide definitions for each taxonomy category shown in Fig. 10. Additionally, we explain the specific metrics used to assess data heterogeneity within each category.

- 1) Label skew : These metrics quantify the imbalance in class distributions across clients. This category is divided into Distribution-based, Statistical Test-based, and Class-based.
- a) Distribution-based : These metrics compare label distributions between clients or against a global distribution. Label distributions refer to the proportion of different classes within a dataset. The comparison uses widely known distances and divergences like Earth Mover's distance (also known as Wasserstein distance) [143], [159]-[161], Hellinger distance [128], [162], Jensen-Shannon divergence [163], [164], Kullback-Leibler divergence [165], and similarities like the Cosine [40], [146] and Jaccard [166].

Despite their utility, the previous metrics exhibit limitations when applied to non-IID data distributions in FL. First, distance-based metrics like Earth mover's distance (Wasserstein) and Hellinger Distance, while effective in capturing dis- similarities, can become computationally expensive, especially in high-dimensional settings [167]. Furthermore, divergences such as Jensen-Shannon and Kullback-Leibler can be sensitive to small sample sizes, which may lead to inaccurate results when comparing distributions with limited data among clients. Lastly, similarity-based metrics like Cosine only consider the angle between two vectors, ignoring their magnitudes. The latter might result in misleading comparisons of label distributions between clients, especially when the data is highly imbalanced [168].

b) Statistical Test-based : These metrics leverage statistical tests to assess the significance of label distribution differences. It includes widely known tests such as the KolmogorovSmirnov statistic [169] and the Chi-squared test [170]. Another alternative test is the Maximum Mean Discrepancy. The latter compares distributions by identifying a continuous function f , calculating its mean value for each distribution, and measuring the discrepancy between these means. This discrepancy reflects the distributions' differences, with larger discrepancies indicating greater differences.

- c) Class-based : These metrics focus on quantifying the non-IID ness based on the volume of participation of one or more classes compared to the maximum number of classes per client. Considering that fewer classes per client mean uneven data distribution and large heterogeneity of the dataset, Zawad et al. [170] proposed the Heterogeneity Index ( HI ) as defined in Eq. 8:

$$
H I = 1 - \frac { 1 } { ( C _ { \max } - 1 ) } \cdot ( c - 1 ) \quad ( 8 )
$$

where c is the maximum number of classes per client, and C max is the total number of classes in the dataset. Notice that HI may oversimplify the complexity of data distributions, failing to capture nuanced variations in client data [171].

The Imbalance Ratio ( IR ) [172], [173] is another metric proposed in the literature defined as in Eq. 9.

$$
I R ( \xi ) = \frac { \max _ { i } \xi _ { i } } { \min _ { j } \xi _ { j } }
$$

where ξ i represents the frequency of class i in the dataset. This metric measures the imbalance between the majority and minority classes. A higher IR value indicates a greater class imbalance in the dataset. The IR has two main limitations: it is primarily suited for binary classification problems and may oversimplify multi-class imbalances by ignoring intermediate class distributions.

- 2) Attribute skew : These metrics measure the differences in feature distributions across clients. This category includes only metrics using the attributes distribution per client ( Distribution-based ). It includes measures like Hellinger distance, Earth mover's distance, and Jensen-Shannon divergence, applied to feature spaces rather than label spaces [128].
- 3) Quantity skew : The metrics evaluate the imbalance in data volume across clients. This category includes only metrics leveraged on the client's participation distribution ( Distribution-based ). It often uses measures similar to label and attribute skew (e.g., Hellinger distance, Earth mover's

distance, Jensen-Shannon divergence) but applied to sample counts or proportions [128].

- 4) Label + Attribute skew : These metrics combine label and attribute skew aspects to provide a more comprehensive assessment of data heterogeneity. This category includes Model-based, Encoder-based, and Model Performance-based.

a) Model-based : In this category, the metrics use the model to infer the combined effect of label and attribute skew. The model traveling metric [132] estimates how well a model generalizes across different, often skewed, non-IID data partitions. During training, the model is periodically moved between different data partitions to assess its accuracy on other clients. Comparing the model's performance on its original data partition with its accuracy on a new partition allows for the estimation of accuracy loss, which reflects the degree of non-IIDness.

The other two metrics proposed in the literature are inner variation and outer variation [155]. Inner variation measures the randomness in sampling data from a client's local distribution f ( i ) Y and is defined as E [ g i ] -E [ g 2 i ] , where g i is the gradient of client i . Outer variation captures the difference between a client's local distribution P i and the global distribution 1 | H | ∑ i ∈ H P i , and is given by E [ g i ] -E [ u 2 ] , where u represents the global gradient. In IID settings, outer variation is zero, while in feature and label-skewed scenarios, it becomes nonzero, indicating higher non-IIDness.

Dataset entropy [174] is a metric designed to characterize a dataset's distribution, information quantity, unbalanced structure, and non-IID nature, independently of the models used. It is computed through a generalized clustering strategy, utilizing a custom similarity matrix that integrates features and supervised outputs, making it suitable for classification and regression tasks. The metric's reliance on clustering and a custom similarity matrix can lead to high computational overhead, potentially diminishing its cost-saving benefits in large FL setups.

The universal bound [175] s G is a metric that indicates the extent to which local data distributions differ. A value of s G = 0 signifies that the dataset IID, while higher values indicate increasing levels of non-IID characteristics, reflecting more significant variability among the workers' data. This bound is crucial for analyzing the convergence rates of algorithms like FedAvg , as it provides insights into how non-IID conditions impact the training process and overall performance of FL models.

b) Encoder-based : This type of metric uses learned representations from the data to quantify non-IIDness across both label and attribute dimensions. The only metric in this category The Client-Wise Non-IID Index (CNI) [176]. The CNI measures how different the data distribution of a client i is from that of other clients. It is defined as:

̸

$$
\begin{aligned}
C N ( i ) = \frac { \left \| \left ( \frac { 1 } { | C _ { i } | } \sum _ { k } E n ( \mathcal { D } _ { i } ^ { k } ) \right ) - \left ( \frac { 1 } { | C _ { j } | } \sum _ { l \neq i } E n ( \mathcal { D } _ { j } ^ { l } ) \right ) \right \| _ { 2 } } { \sigma ( E n ( \mathcal { D } ) ) } \quad ( 1 0 ) \quad \text {can help} \quad \begin{array} { c c } \text {clients} \\ \text {The mean} \end{array}
\end{aligned}
$$

where D = ⋃ K i =1 D i , D k i denotes the data belonging to the k -th class in D i . | C i | is the number of classes in D i , σ ( · ) is the standard deviation and is used to normalize the scale, and

∥ · ∥ 2 indicates the ℓ 2 -norm. The intuition behind Eq. 10 is to measure the distance between the average data representations from different classes in feature space on a given client and the counterpart over all the other clients.

c) Performance-based : It refers to metrics that directly measure the impact of combined label and attribute skew on model performance. Under this category the only metric found is Dataskew [158], defined as in Eq. 11:

$$
\ D a t s k e w = \frac { \max ( \Delta A c c u r a c y _ { p a r i w s } ) } { \frac { 1 } { K } \sum _ { i = 1 } ^ { K } A c c u r a c y _ { i } } \quad ( 1 1 )
$$

Where max(∆ Accuracy pairwise ) is the maximum pairwise deviation of accuracy between clients. 1 K ∑ K i =1 Accuracy i is the average accuracy across all clients. A high Dataskew value (close to 1) indicates strong data heterogeneity. It might introduce bias into the calculation as it includes the client on which the initial model was trained. If that client has a significantly different data distribution, it could skew the interpretation of the metric.

- 5) Label + Attribute + Quantity skew : This category represents the most comprehensive metrics for all three main types of non-IIDness. It is noticeable that quantifying more than one skewness simultaneously is paramount to understanding the complete behavior of non-IIDness in the FL system.
- a) Distribution-based : Based on the label, attribute, and quantity distribution distributions, these metrics attempt to capture the combined effects of such types of skew in a single measure. The Four-Dimensional Quantitative Measure (FDQM) [157] measures heterogeneity across four key dimensions:
- Noise Estimation for Feature Distribution Skew : This dimension quantifies the noise or variability in feature distributions across different clients, assessing how features deviate from one another.
- Degree of Difference for Label Distribution Skew : This measures the difference in class label distributions among clients, capturing the extent of label imbalance.
- Degree of Difference for Quantity Skew : This dimension evaluates the variability in the amount of data across clients, identifying differences in dataset sizes.
- Feature Distribution Estimation for Feature Differency : This estimates the difference in feature distributions across clients, focusing on the dissimilarity in feature values or characteristics.

FDQM combines these four dimensions into a single comprehensive metric that helps quantify non-IIDness. FDQM involves multiple statistical measures, such as noise estimation, Dirichlet concentration parameters, and Gaussian Mixture Models. These calculations can become computationally intensive, mainly when applied to large-scale FL systems with many clients. Additionally, it does not have an upper bound that can help to understand 'acceptable' levels of heterogeneity. The metric assumes that client data follows specific statistical distributions. If the client data does not align with these assumptions, the FDQM may provide inaccurate or misleading heterogeneity estimates.

Our survey reveals a notable absence of metrics that can simultaneously measure combinations of data heterogeneities, such as label with quantity skew, attribute with quantity skew, or combinations with spatiotemporal heterogeneity, participation skew, and modality skew. This measurement gap is particularly concerning, given that these heterogeneities coexist in real-world FL scenarios. Without combined metrics, researchers and practitioners lack the tools to effectively quantify and understand how multiple types of data heterogeneity jointly impact model performance.

## VI. NON-IID SOLUTIONS IN FL

This section briefly discusses the leading solutions for nonIID datasets in FL to tackle the downstream effects exposed in Section II. Notice that most of these solutions focus on one type of non-IIDness, which is something future research should improve. In particular, we describe in the following paragraphs the solutions that are employed the most in the selected papers of our survey (see Fig. 11).

Fig. 11: Relative prevalence of state-of-the-art solutions to tackle non-IIDness

<!-- image -->

FedProx [177] . Designed to combat heterogeneity within FL environments, addressing both system and statistical heterogeneity. As an extension and reparameterization of the FedAvg method, FedProx introduces flexibility by enabling each participating device to perform varying amounts of work, adhering to individual device-level system constraints. The advantages of FedProx include providing convergence guarantees even when learning from non-IID data, thus mitigating the effects of statistical heterogeneity. Moreover, it offers the advantage of requiring only minor adjustments to existing methods, making it straightforward to implement and integrate into preexisting FL systems. However, it is worth noting that FedProx may still present some limitations. For example, its performance can be sensitive to the choice of the proximal term, and it may not fully address all forms of statistical heterogeneity, particularly in scenarios with extreme distribution differences.

Scaffold [178] . It is an algorithm to mitigate the challenges posed by data heterogeneity among clients in an FL setting. This method leverages control variates, a variance reduction technique, to counteract the client drift phenomenon often occurring in local updates during the FL process. The critical advantage of Scaffold lies in its ability to address the performance issues stemming from non-IID client data. Through the strategic use of control variates, Scaffold effectively corrects the local update drift, enhancing FL's convergence and overall effectiveness. One potential constraint is the need for adequate domain knowledge and expertise in configuring the control variates, which may pose challenges in specific practical scenarios.

Data sharing [27], [179], [180] . Contrary to the standard FL approaches, where the server and the clients do not share data, this approach envisions that initially, the server trains a global model on a globally shared dataset. A fraction of this dataset is delivered to the clients that update their local models, employing the local private training data and the shared global data. Experiments show that accuracy can be increased by 30% for the CIFAR-10 dataset with only 5% globally shared data. Despite providing good performance, this method suffers from some shortcomings: it is difficult to get an initial goodquality global dataset since the server has no idea about the data distributions among the connected clients, and sharing data violates the requirement of privacy-preserving learning, which is a fundamental motivation of FL.

FedNova [181] . FedNova is a normalized averaging method that targets resolving the objective inconsistency issue commonly encountered in FL. It eliminates objective inconsistency while maintaining rapid error convergence, making it a valuable addition to the FL landscape. The advantages of FedNova include its ability to provide a systematic understanding of the solution bias and convergence slowdown resulting from objective inconsistency in FL. While it addresses objective inconsistency, its performance and applicability can still be influenced by the specific characteristics of the FL problem, including the degree of heterogeneity among the clients' data distributions.

Client clustering [66], [92], [182]-[184] . Most FL approaches assume that the whole system distills only one global model. However, this may be a challenging and unrealistic assumption, especially in heterogeneous data environments. In client clustering, this assumption is relaxed. Instead of training a single FL model among all clients, clients form clusters characterized by more balanced and homogeneous data and then perform FL within the individual clusters. This approach results in more performant intra-cluster models.

The clusters should be created without disclosing sensitive client information to the central server to preserve privacy. To this purpose, the literature proposes two main methods: the similarity of the loss values and the similarity of model weights. In the former approach, the server constructs multiple global models instead of a single one and shares them with the clients. Then, the clients can compute a local empirical loss and update the received cluster model with the smallest loss value. Finally, the updated cluster model is returned to the server for cluster model aggregation. In the latter approach, an initial global model is downloaded to each device to update the weights locally, which are returned to the server. The server can finally derive the similarity scores employing received model weights and group the clients into clusters given such scores [41].

Model-contrastive learning (MOON) [185] . It uses the similarity between model representations to correct the local training of individual parties, i.e., conducting contrastive learning at the model level. The key idea of contrastive learning is to reduce the distance between the representations of different augmented views of the same image (i.e., positive pairs) and increase the distance between the representations of augmented views of different images (i.e., negative pairs). Specifically, MOON corrects the local updates by maximizing the agreement of representation learned by the current local model and the representation learned by the global model. Extensive experiments show that MOON significantly outperforms the other state-of-the-art FL algorithms on various image classification tasks.

Knowledge Distillation (KD) [186]-[190] . It is a technique where a larger, more complex model, known as the 'teacher,' transfers its knowledge compactly and efficiently to smaller, simpler models, referred to as the 'student.' The compact and efficient way to transfer knowledge is soft labels, which represent the probabilities the teacher model assigns to each class. The latter conveys not just the final decision but also the confidence levels across potential outcomes. The goal is to transfer the comprehensive knowledge of the teacher model to a student model that retains much of the teacher's accuracy and performance but with significantly fewer parameters. This typical teacher-student framework in KD is quite suitable for FL. The idea is that well-performing models (teachers) can transfer their knowledge to poorly performing ones (students). KD-based FL algorithms only require clients to exchange their local models' soft labels (e.g., logits) without uploading model parameters or data, thus significantly reducing potential privacy risks and communication costs.

Data Augmentation [191]-[194] . It has been employed to increase the diversity of training data by applying suitable (e.g., random but realistic) transformations to the original dataset. The same technique can mitigate local data imbalance issues in FL using three main data augmentation methods: vanilla, mixup, and generative adversarial network (GAN). In vanilla data augmentation, each client sends its label distribution information to the server. The server computes the global statistics, considering the information from each client, and informs the client on the degree of augmentation needed to mitigate the imbalance. The original local data sample and the augmented data are used together to update the local model parameters.

The core idea of mixup is to synthesize new samples by combining existing data points. To preserve privacy, the existing data points are encoded and delivered to the server that performs the mixup operation. The obtained balanced dataset is used to train a global model that is then delivered to the clients until the training converges.

Federated GANs are designed to train a generator in the presence of non-IID data. The generator is trained in the server, employing the data shared by the clients, and it is finally delivered to the clients. They can eventually use it to replenish the training data for the missing components (e.g., labels) to obtain a more balanced local dataset. Despite being proven effective, most of these techniques rely on data sharing, which might increase the risk of data privacy leakage.

Dynamic Client Selection [40], [42], [160], [195]-[197] . Most FL papers assume unbiased client participation, where clients are selected randomly or in proportion to their data sizes. In Dynamic Client Selection, clients are selected employing a bias strategy. In other words, clients are dynamically chosen at each round to achieve a good trade-off between convergence speed and solution bias. Its advantages include faster convergence by prioritizing clients with higher-quality data, better handling system heterogeneity, and efficient use of resources. However, it may introduce selection bias, leading to over-reliance on specific clients, increased bias, and decreased model generalization. Additionally, it can increase the complexity of client selection, add communication overhead, and risk underrepresented clients with limited computational resources.

FedDyn [198] This method introduces a novel approach to construct a proxy dataset and extract local knowledge dynamically. Instead of using the average strategy, they implemented focus distillation to emphasize reliable knowledge, effectively addressing the non-IID issue where local models may have biased knowledge. The average strategy weakens knowledge quality by treating reliable and unreliable knowledge equally. Additionally, they applied local differential privacy techniques on the client side to safeguard private user information from local knowledge. Experimental results demonstrated that their method achieves faster convergence and reduces communication overhead compared to baseline approaches.

In FL, the choice between personalized and a single global model significantly impacts how heterogeneity is addressed. A global model simplifies collaboration but struggles with performance when client data distributions differ significantly, often requiring advanced optimization techniques like FedProx or Scaffold to mitigate these effects. In contrast, personalized models (a.k.a personalized FL [199]) allow each client to tailor the model to their local data, improving accuracy for clients with skewed or specialized datasets but adding complexity and computational cost [200]. Techniques like client clustering, KD, or dynamic client selection aim to balance personalization with shared knowledge. The decision between personalized models and a global model ultimately depends on factors such as the degree of data heterogeneity across clients, the need for model consistency, the application's tolerance for varying performance across clients, and the available computational resources for handling the added complexity of personalization [201], [202].

It is worth noting that Fig. 11 illustrates the prevalence of each solution discussed in this section. From the papers retrieved, we can see that FedProx is the most commonly employed solution, accounting for nearly 25% of the total participation. In contrast, solutions like Scaffold, Data Sharing, and FedNova are utilized, but their participation rates are significantly lower than those of FedProx.

## VII. FRAMEWORKS AND TOOLS FOR HETEROGENEOUS DATA IN FL

FL has witnessed the development of numerous standardized frameworks to simplify its implementation, including partition protocols to federate centralized data and tackle the challenges of non-IID scenarios. This section comprehensively reviews notable FL frameworks, emphasizing their distinctive characteristics for partitioning, quantifying, and handling nonIID data.

FEDML [203] Also known as TensorOpera AI, it offers a comprehensive FL research and deployment platform, supporting various partition protocols and non-IID solutions. Flower [154] provides a flexible framework for FL, emphasizing partition protocols in their Flower Datasets library. TensorFlow Federated (TFF) [204] integrates FL capabilities into the TensorFlow ecosystem, offering tools for simulating simple FL scenarios. PySyft [205], while limited in partition protocols, focuses on privacy-preserving ML techniques. OpenFL [206] aims to standardize FL workflows across different industries, particularly healthcare and finance. FLUTE [207] specializes in FL for natural language processing tasks, addressing unique challenges in text data. Federated AI Technology Enabler (FATE) [208] provides a comprehensive suite for secure FL, emphasizing privacy and security in collaborative AI. Lastly, FedLab [209] supports various partition protocols, offering a robust environment for researching heterogeneous data scenarios in FL.

In addition to standardized frameworks, tools and benchmarks have been developed to facilitate research and evaluation in FL with non-IID data. FedArtML [128] is a comprehensive toolkit explicitly designed for exploring and analyzing the impact of non-IID data in FL scenarios, offering a range of partition protocols and non-IID metrics. LEAF [210] provides a modular benchmarking framework for learning in federated settings, featuring a collection of open-source datasets and a suite of evaluation tools to assess FL algorithms under various data distribution scenarios. NIID-Bench NIID-Bench [211] provides a standardized benchmarking platform, providing researchers with a consistent environment to compare different approaches and assess their effectiveness in handling data heterogeneity.

Table II comprehensively compares various standardized frameworks and tools, focusing on their capabilities in partitioning centralized data into non-IID federated data and tackling non-IID issues. The comparison is divided into three main categories: Partition protocols, non-IID metrics, and non-IID solutions. Regarding partition protocols, FedLab, FedArtML, and LEAF appear to be the most comprehensive, supporting label, attribute, and quantity skewness. Most frameworks support label skew, but attribute skew is less commonly supported. Notably, PySyft seems to lack support for any partition protocols.

Regarding non-IID metrics, FedArtML is the only framework with comprehensive support for label, attribute, and quantity skew metrics. Most frameworks have limited or no support for non-IID metrics, and no framework fully supports metrics for combined skews (label + attribute or label + at- tribute + quantity). For non-IID solutions, FEDML and Flower appear to be the most versatile. FedProx is universally supported across all applicable frameworks, while more advanced solutions like MOON, data augmentation, and dynamic client selection are less commonly supported. There is a general lack of support for non-IID metrics and combined skew scenarios across most frameworks.

Fig. 12: Relative prevalence of use for standardized frameworks in FL

<!-- image -->

Fig. 12 illustrates the adoption rates of standardized frameworks for FL across the research papers studied. It reveals FEDML as the clear leader, utilized in approximately 30% of papers. Flower is the second most popular framework, appearing in about 20% of studies. TFF and PySyft show similar usage rates, each featuring in roughly 18-19% of papers. Lastly, OpenFL, FLUTE, and FATE exhibit more limited use, accounting for less than 5% of the articles.

While reviewing the literature retrieved, we realized that, surprisingly, only 14.2% of them use FL frameworks to implement their proposals . This low adoption rate suggests that a substantial portion of the research in FL is conducted in isolation, potentially relying on custom or ad-hoc implementations rather than established frameworks designed to facilitate FL experiments. The latter disconnect may hinder the reproducibility and comparability of results, as custom implementations can vary widely in terms of functionality and efficiency. Encouraging greater integration of FL frameworks could lead to more consistent experimental practices, foster collaboration, and accelerate advancements by leveraging these established tools' robust features and optimizations.

## VIII. ESSENTIAL FINDINGS AND KEY INSIGHTS

This section highlights the essential findings and critical insights obtained through our survey of the current literature on non-IID data in FL. We aim to provide practitioners with robust conclusions and thought-provoking considerations for developing practical solutions in this challenging area of FL. These lessons encompass various aspects, including the quantification and classification of non-IID-ness, the interplay between different types of data skew, and the practical implications for implementation and standardized framework selection.

TABLE II: FL standardized frameworks and tools for non-IID data comparison ( /check-circle : Sufficient, /minus-circle : Under development/incomplete, /times-circle : Unknown/Not implemented), -: Not applicable

|                                     | FEDML         | Flower        | TFF           | PySyft        | OpenFL        | FLUTE         | FATE          | FedLab        | FedArtML      | LEAF          | NIID-Bench    |
|-------------------------------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| Partition protocols                 | /minus-circle | /check-circle | /minus-circle | /times-circle | /minus-circle | /minus-circle | /minus-circle | /check-circle | /check-circle | /check-circle | /minus-circle |
| ★ Label skew                        | /minus-circle | /check-circle | /minus-circle | /times-circle | /minus-circle | /check-circle | /minus-circle | /check-circle | /check-circle | /check-circle | /check-circle |
| ★ Attribute skew                    | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /check-circle | /check-circle | /check-circle | /minus-circle |
| ★ Quantity skew                     | /minus-circle | /check-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /check-circle | /check-circle | /minus-circle | /minus-circle |
| non-IID metrics                     | /times-circle | /minus-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /minus-circle | /times-circle | /times-circle |
| ★ Label skew                        | /times-circle | /minus-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /check-circle | /times-circle | /times-circle |
| ★ Attribute skew                    | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /check-circle | /times-circle | /times-circle |
| ★ Quantity skew                     | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /check-circle | /times-circle | /times-circle |
| ★ Label + Attribute skew            | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle |
| ★ Label + Attribute + Quantity skew | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle |
| non-IID solutions                   | /check-circle | /check-circle | /times-circle | /minus-circle | /times-circle | /minus-circle | /times-circle | /check-circle | -             | -             | -             |
| ★ FedProx                           | /check-circle | /check-circle | /check-circle | /check-circle | /check-circle | /check-circle | /check-circle | /check-circle | -             | -             | -             |
| ★ SCAFFOLD                          | /check-circle | /check-circle | /times-circle | /check-circle | /times-circle | /check-circle | /times-circle | /check-circle | -             | -             | -             |
| ★ Data sharing                      | /times-circle | /minus-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | -             | -             | -             |
| ★ FedNova                           | /check-circle | /check-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /check-circle | -             | -             | -             |
| ★ Client clustering                 | /times-circle | /minus-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /check-circle | -             | -             | -             |
| ★ MOON                              | /times-circle | /check-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | -             | -             | -             |
| ★ Knowledge distillation            | /check-circle | /check-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | -             | -             | -             |
| ★ Data augmentation                 | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | -             | -             | -             |
| ★ Dynamic client selection          | /times-circle | /minus-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /minus-circle | -             | -             | -             |
| ★ FedDyn                            | /check-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /times-circle | /check-circle | -             | -             | -             |

- 1) Lack of consensus on non-IID classification and quantification . Researchers have no agreement on how to classify and quantify non-IID-ness in FL. This lack of standardization makes it challenging to compare different studies and solutions. Furthermore, no consensus exists on sufficient conditions or scenarios (regarding data partition and simulations) to test new solutions addressing non-IIDness adequately.
- 2) Importance of quantifying non-IID-ness . Among the studied papers, just 13.1% of them employed metrics to quantify the level of non-IIDness [175], [212]-[215]. Using such metrics to quantify non-IID-ness is crucial for adequately characterizing FL datasets. Without such quantification, assessing the severity of non-IID issues and the effectiveness of proposed solutions becomes difficult.
- 3) Interdependence of skew types . When partitioning datasets to create a specific type of skew (e.g., label skew), researchers often inadvertently introduce side effects in other types of skewness. Nevertheless, they do not take into account such effects during their experimentation. For example, in the research papers reviewed, 60.3% of the papers that include label skew approaches do not mention anything about quantity skew, even if the partition methods employed generate the mentioned side effect [95], [216][224]. This interdependence highlights the complexity of non-IID scenarios and the need for a more holistic approach to dataset preparation and analysis.
- 4) Prevalence of label skew studies . Our survey indicates that a majority of papers focus on label skew. While this is an important aspect of non-IID data, causing the major deterioration of FL models [225]-[230], the disproportionate attention may lead to overlooking other critical types of non-IID skews.
- 5) Limitations of single skew analysis . Quantifying only one skew type at a time is insufficient for fully understanding and addressing non-IID-ness in FL. In real-life scenarios, the data is generally affected by more than one type of skewness, and the simulations performed should mimic such settings. Thus, a more comprehensive approach that simultaneously considers multiple types of skewness is necessary to develop robust solutions.
- 6) Custom FL implementations vs. standardized frameworks . Only 14.2% of the reviewed papers employed standardized FL frameworks (i.e., FEDML, Flower, Pysyft, etc.) to implement their solutions to tackle non-IIDness. While this approach offers flexibility, it also introduces risks such as implementation errors and reduced reproducibility.
- 7) Few works in multimodality skew research . The multimodality skew, as exposed in Section IV, is a topic that has not been formally included in the previous surveys for non-IIDness. Despite its relevance, the research papers that include that topic are just a few [87]-[89], [216], [231][234]. As real-world FL applications often involve diverse data types and modalities, understanding and addressing multimodality skew is a key aspect for practical implementations with the aim of understanding and tackling the consequences of non-IID data in FL.

## IX. FUTURE DIRECTIONS AND TRENDS

This section delineates promising future research directions and emerging trends for developing innovative solutions to address non-IID data challenges in FL. We aim to highlight key areas that can guide researchers in formulating novel approaches and methodologies to enhance our understanding of the impact of non-IIDness on FL systems.

- 1) Evaluation and benchmarking . A critical area for future research in non-IID FL is the development of more

- sophisticated evaluation and benchmarking methodologies. Real-life FL data is typically affected by multiple types of skewness simultaneously, a nuance often overlooked in current simulations [107], [235]-[239]. Future research should focus on developing comprehensive partition protocols that consider various forms of data heterogeneity concurrently, such as label skew, feature skew, and quantity skew simultaneously. This enhanced evaluation framework will lead to more reliable performance assessments and drive the creation of FL algorithms that are more resilient and adaptable to diverse non-IID scenarios.
- 2) Theoretical advancements . Theoretical advancements represent a crucial frontier in addressing non-IID challenges in FL. A primary focus should be the development of new, more sophisticated metrics for quantifying data heterogeneity. Current measures often fail to capture the full complexity of non-IID scenarios, particularly when multiple types of data skewness coexist. Future research should aim to create comprehensive metrics that can accurately reflect the multidimensional nature of data heterogeneity, incorporating as many data skews as possible.

Additionally, significant effort should be directed toward improving generalization bounds for non-IID FL. The development of tighter bounds on generalization performance is essential for bridging the gap between theoretical guarantees and practical performance. The latter refers to theoretical guarantees or limits on how well a model trained on a distributed dataset can perform on unseen data [240][243]. These improved bounds should account for various aspects of data heterogeneity and federation strategies, providing more accurate model performance predictions across diverse non-IID settings. Such theoretical advancements will leverage the development of more robust and efficient algorithms capable of maintaining high performance in severe data heterogeneity.

- 3) Modality skew research . A promising avenue for future research in FL is the exploration of multimodal learning techniques to address the increasingly complex nature of real-world data. As applications of FL expand across diverse domains, there is a growing need to develop robust methods capable of handling multiple data modalities simultaneously, such as text, images, audio, and sensor data [90], [244]. Future research should focus on creating novel FL algorithms that can effectively integrate and learn from these heterogeneous data types while preserving privacy and maintaining efficiency. This direction presents unique challenges in the context of non-IID data, as different modalities may exhibit varying degrees and types of heterogeneity across clients.

Therefore, it is vital to develop new metrics to accurately quantify multimodality skew, capturing the distribution differences within each modality and the relationships and dependencies between different modalities. These metrics should be capable of measuring aspects such as crossmodal correlation discrepancies, modality-specific feature disparities, and variations in the relative importance of different modalities across clients. The latter helps to get the full potential of diverse, real-world datasets while

- effectively managing the complexities of non-IID data distributions.
- 4) Addressing data heterogeneity impact . Addressing data heterogeneity remains a central challenge in FL, necessitating innovative approaches to enhance model performance and adaptability in non-IID settings [79], [245], [246]. Future research should prioritize the development of more effective algorithms designed to handle diverse forms of non-IIDness, specifically under high levels of non-IIDness. These advanced algorithms should aim to mitigate the negative impacts of non-IID data on model convergence, bias, and generalization (robustness).
- Additionally, there is a pressing need for adaptive methods capable of dynamically adjusting to different nonIID scenarios encountered in real-world applications. Such methods could leverage online learning techniques or metalearning approaches to rapidly identify and adapt to varying degrees and types of data heterogeneity across clients or over time. Furthermore, the emerging field of Federated Neural Architecture Search (FNAS) presents a promising direction for automatically designing optimal model architectures in federated settings, particularly under non-IID conditions [247]-[250]. FNAS could potentially uncover novel network structures that are inherently more robust to data heterogeneity, leading to improved performance and efficiency in diverse FL scenarios.
- 5) Communication efficiency . Enhancing communication efficiency remains another critical area for future research in FL. As the scale and complexity of federated systems grow, the need for optimized communication protocols becomes increasingly paramount. Future research should focus on designing advanced compression techniques tailored to model updates in non-IID environments [251][256]. These techniques must strike a delicate balance between reducing communication overhead and preserving the integrity of model updates, which is especially challenging when clients have disparate data distributions. Innovative approaches might include adaptive compression methods that adjust their strategies based on the degree of data heterogeneity or the importance of specific model parameters.

Exploring asynchronous communication protocols presents another promising avenue for improving efficiency and scalability in FL systems. Asynchronous methods could allow for more flexible participation of clients, potentially mitigating some of the challenges posed by non-IID data distributions [139], [257]. However, careful consideration must be given to ensuring model convergence and managing the potential staleness of updates in asynchronous settings. The latter enables the deployment of FL in a broader range of real-world scenarios with varying network conditions and client capabilities.

- 6) Model generalization and task adaptation . A final area for future research in FL is enhancing model generalization and task adaptation in non-IID data. As real-world data distributions often evolve, there is a pressing need to investigate methods to improve model robustness and adaptability to these dynamic scenarios [111], [258]-[263].

Future work should focus on developing techniques to mitigate catastrophic forgetting in non-IID settings, where new data distributions may cause the model to lose performance on previously learned tasks rapidly. The latter could involve exploring continual learning approaches or developing novel regularization techniques specifically designed for federated environments with heterogeneous and time-varying data.

While much of the current research in FL has centered on image classification tasks, there is a significant opportunity to extend these methods to a broader range of domains and applications. Future research should aim to adapt and optimize FL algorithms for diverse tasks such as natural language processing, speech recognition, and time series analysis [264]-[268]. This expansion will require addressing the challenges posed by non-IID data in these domains, including handling variable-length inputs, dealing with domain-specific data skew, and managing the increased complexity of model architectures often required for these tasks.

- 7) Public datasets with non-IIDness quantification . A crucial step toward advancing research on non-IID data in FL is the creation and dissemination of publicly available datasets with explicit and standardized characterizations of their non-IID properties. Including clear non-IID metrics to quantify the extent and types of non-IIDness in these datasets would facilitate benchmarking, reproducibility, and the development of more robust methods tailored to diverse real-world scenarios.

## X. CONCLUSION

In summary, this work offers a comprehensive technical survey of the state-of-the-art non-IID data in FL. Unlike existing surveys, we include a detailed taxonomy for nonIID data, partition protocols and metrics to quantify nonIIDness, popular solutions to address data heterogeneity, and standardized frameworks employed in FL with heterogeneous data. We include modality skew as a novel category in the taxonomy data heterogeneity, which appears as a new trend among the researchers. We indicate a lack of consensus on non-IID classification and quantification, highlighting the importance of using metrics to measure non-IIDness. We claim that quantifying only one skew type at a time is insufficient for fully understanding and addressing non-IIDness. Finally, we suggest some research directions that can be adopted in future investigations in the field.

## XI. ACKNOWLEDGMENTS

Daniel Mauricio Jimenez G. and Andrea Vitaletti were partially supported by PNRR351 TECHNOPOLE - NEXT GEN EU Roma Technopole -Digital Transition, FP2 -Energy transition and digital transition in urban regeneration and construction and Sapienza Ateneo Research grant 'La disintermediazione della Pubblica Amministrazione: il ruolo della tecnologia blockchain e le sue implicazioni nei processi e nei ruoli della PA.' Aris Anagnostopoulos was supported by the ERC Advanced Grant 788893 AMDROMA, the EC H2020RIA project 'SoBigData++' (871042), the PNRR MUR project PE0000013-FAIR, the PNRR MUR project IR0000013-SoBigData.it, and the MUR PRIN project 2022EKNE5K 'Learning in Markets and Society.' Ioannis Chatzigiannakis was supported by PE07-SERICS (Security and Rights in the Cyberspace) -European Union NextGeneration-EU-PE0000014 (Piano Nazionale di Ripresa e Resilienza - PNRR). Andrea Vitaletti was supported by PE11 - MICS (Made in Italy - Circular and Sustainable) - European Union Next-Generation-EU (Piano Nazionale di Ripresa e Resilienza - PNRR) and the project SERICS (PE00000014) under the MUR National Recovery and Resilience Plan funded by the European Union - NextGenerationEU.

Telef´ onica's contribution was partially supported by The Ministry of Economic Affairs and Digital Transformation of Spain and the European UnionNextGenerationEU programme for the 'Recovery, Transformation and Resilience Plan' and the 'Recovery and Resilience Mechanism' under agreements TSI-063000-2021-142 and TSI-063000-2021147 (6G-RIEMANN). Additionally, this research has been as well partially supported by funding received from the Smart Networks and Services Joint Undertaking (SNS JU) under the European Union's Horizon Europe research and innovation programme under Grant Agreement No 101096435 (CONFIDENTIAL6G).

## REFERENCES

- [1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, 'Communication-efficient learning of deep networks from decentralized data,' in Artificial intelligence and statistics . PMLR, 2017, pp. 1273-1282.
- [2] D. Chai, L. Wang, L. Yang, J. Zhang, K. Chen, and Q. Yang, 'A survey for federated learning evaluations: Goals and measures,' IEEE Transactions on Knowledge and Data Engineering , 2024.
- [3] P. Ullagaddi, 'Gdpr: Reshaping the landscape of digital transformation and business strategy,' International Journal of Business Marketing and Management , vol. 9, no. 2, pp. 29-35, 2024.
- [4] S. Lincke, 'Complying with hipaa and hitech,' in Information Security Planning: A Practical Approach . Springer, 2024, pp. 345-365.
- [5] D. M. Jimenez Gutierrez, H. M. Hassan, L. Landi, A. Vitaletti, and I. Chatzigiannakis, 'Application of federated learning techniques for arrhythmia classification using 12-lead ecg signals,' in International Symposium on Algorithmic Aspects of Cloud Computing . Springer, 2023, pp. 38-65.
- [6] T. Awosika, R. M. Shukla, and B. Pranggono, 'Transparency and privacy: the role of explainable ai and federated learning in financial fraud detection,' IEEE Access , 2024.
- [7] D. Deng, X. Wu, T. Zhang, X. Tang, H. Du, J. Kang, J. Liu, and D. Niyato, 'Fedasa: A personalized federated learning with adaptive model aggregation for heterogeneous mobile edge computing,' IEEE Transactions on Mobile Computing , 2024.
- [8] A. Krizhevsky, G. Hinton et al. , 'Learning multiple layers of features from tiny images,' 2009.
- [9] Y. LeCun, C. Cortes, and C. Burges, 'Mnist handwritten digit database,' ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist , vol. 2, 2010.
- [10] H. Xiao, K. Rasul, and R. Vollgraf, 'Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms,' arXiv preprint arXiv:1708.07747 , 2017.
- [11] A. Mora, A. Bujari, and P. Bellavista, 'Enhancing generalization in federated learning with heterogeneous data: A comparative literature review,' Future Generation Computer Systems , 2024.
- [12] Z. Lu, H. Pan, Y. Dai, X. Si, and Y. Zhang, 'Federated learning with non-iid data: A survey,' IEEE Internet of Things Journal , 2024.
- [13] J. Pei, W. Liu, J. Li, L. Wang, and C. Liu, 'A review of federated learning methods in heterogeneous scenarios,' IEEE Transactions on Consumer Electronics , 2024.

- [14] C. Chen, T. Liao, X. Deng, Z. Wu, S. Huang, and Z. Zheng, 'Advances in robust federated learning: Heterogeneity considerations,' arXiv preprint arXiv:2405.09839 , 2024.
- [15] M. F. Criado, F. E. Casado, R. Iglesias, C. V. Regueiro, and S. Barro, 'Non-iid data and continual learning processes in federated learning: A long road ahead,' Information Fusion , vol. 88, pp. 263-280, 2022.
- [16] X. Ma, J. Zhu, Z. Lin, S. Chen, and Y. Qin, 'A state-of-the-art survey on solving non-iid data in federated learning,' Future Generation Computer Systems , vol. 135, pp. 244-258, 2022.
- [17] H. Zhu, J. Xu, S. Liu, and Y. Jin, 'Federated learning on non-iid data: A survey,' Neurocomputing , vol. 465, pp. 371-390, 2021.
- [18] T. Sun, D. Li, and B. Wang, 'Decentralized federated averaging,' IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 45, no. 4, pp. 4289-4301, 2022.
- [19] P. M. Mammen, 'Federated learning: Opportunities and challenges,' arXiv preprint arXiv:2101.05428 , 2021.
- [20] C. Huang, J. Huang, and X. Liu, 'Cross-silo federated learning: Challenges and opportunities,' arXiv preprint arXiv:2206.12949 , 2022.
- [21] D. C. Nguyen, M. Ding, P. N. Pathirana, A. Seneviratne, J. Li, and H. V. Poor, 'Federated learning for internet of things: A comprehensive survey,' IEEE Communications Surveys &amp; Tutorials , vol. 23, no. 3, pp. 1622-1658, 2021.
- [22] L. Liu, J. Zhang, S. Song, and K. B. Letaief, 'Client-edge-cloud hierarchical federated learning,' in ICC 2020-2020 IEEE international conference on communications (ICC) . IEEE, 2020, pp. 1-6.
- [23] C. Zhang, Y. Xie, H. Bai, B. Yu, W. Li, and Y. Gao, 'A survey on federated learning,' Knowledge-Based Systems , vol. 216, p. 106775, 2021.
- [24] S. Banabilah, M. Aloqaily, E. Alsayed, N. Malik, and Y. Jararweh, 'Federated learning review: Fundamentals, enabling technologies, and future applications,' Information processing &amp; management , vol. 59, no. 6, p. 103061, 2022.
- [25] Y. Liu, Y. Kang, T. Zou, Y. Pu, Y. He, X. Ye, Y. Ouyang, Y.-Q. Zhang, and Q. Yang, 'Vertical federated learning: Concepts, advances, and challenges,' IEEE Transactions on Knowledge and Data Engineering , 2024.
- [26] A. G. Roy, S. Siddiqui, S. P¨ olsterl, N. Navab, and C. Wachinger, 'Braintorrent: A peer-to-peer environment for decentralized federated learning,' arXiv preprint arXiv:1905.06731 , 2019.
- [27] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, 'Federated learning with non-iid data,' arXiv preprint arXiv:1806.00582 , 2018.
- [28] K. Maharana, S. Mondal, and B. Nemade, 'A review: Data preprocessing and data augmentation techniques,' Global Transitions Proceedings , vol. 3, no. 1, pp. 91-99, 2022.
- [29] J. G. Avelino, G. D. Cavalcanti, and R. M. Cruz, 'Resampling strategies for imbalanced regression: a survey and empirical analysis,' Artificial Intelligence Review , vol. 57, no. 4, p. 82, 2024.
- [30] G. Varoquaux and O. Colliot, 'Evaluating machine learning models and their diagnostic value,' Machine learning for brain disorders , pp. 601-630, 2023.
- [31] S. A. Rahman, H. Tout, C. Talhi, and A. Mourad, 'Internet of things intrusion detection: Centralized, on-device, or federated learning?' IEEE Network , vol. 34, no. 6, pp. 310-317, 2020.
- [32] Y. Zhang, H. Chen, Z. Lin, Z. Chen, and J. Zhao, 'Fedac: A adaptive clustered federated learning framework for heterogeneous data,' arXiv preprint arXiv:2403.16460 , 2024.
- [33] H. Wu and P. Wang, 'Node selection toward faster convergence for federated learning on non-iid data,' IEEE Transactions on Network Science and Engineering , vol. 9, no. 5, pp. 3099-3111, 2022.
- [34] Q. Li, Y. Diao, Q. Chen, and B. He, 'Federated learning on non-iid data silos: An experimental study,' in 2022 IEEE 38th international conference on data engineering (ICDE) . IEEE, 2022, pp. 965-978.
- [35] H. Lin, J. Lou, L. Xiong, and C. Shahabi, 'Semifed: Semi-supervised federated learning with consistency and pseudo-labeling,' arXiv preprint arXiv:2108.09412 , 2021.
- [36] S. Rai, A. Kumari, and D. K. Prasad, 'Client selection in federated learning under imperfections in environment,' AI , vol. 3, no. 1, pp. 124-145, 2022.
- [37] M. Asad, A. Moustafa, T. Ito, and M. Aslam, 'Evaluating the communication efficiency in federated learning algorithms,' in 2021 IEEE 24th International Conference on Computer Supported Cooperative Work in Design (CSCWD) . IEEE, 2021, pp. 552-557.
- [38] C. Ma, J. Li, M. Ding, H. H. Yang, F. Shu, T. Q. Quek, and H. V. Poor, 'On safeguarding privacy and security in the framework of federated learning,' IEEE network , vol. 34, no. 4, pp. 242-248, 2020.
- [39] G. Xia, J. Chen, C. Yu, and J. Ma, 'Poisoning attacks in federated learning: A survey,' IEEE Access , vol. 11, pp. 10 708-10 722, 2023.
- [40] J. Zhang, J. Wang, Y. Li, F. Xin, F. Dong, J. Luo, and Z. Wu, 'Addressing heterogeneity in federated learning with client selection via submodular optimization,' ACM Transactions on Sensor Networks , vol. 20, no. 2, pp. 1-32, 2024.
- [41] P. Tian, W. Liao, W. Yu, and E. Blasch, 'Wscc: A weight-similaritybased client clustering approach for non-iid federated learning,' IEEE Internet of Things Journal , vol. 9, no. 20, pp. 20 243-20 256, 2022.
- [42] S. R. Pandey, L. D. Nguyen, and P. Popovski, 'A contribution-based device selection scheme in federated learning,' IEEE Communications Letters , vol. 26, no. 9, pp. 2057-2061, 2022.
- [43] P. R. Silva, J. Vinagre, and J. Gama, 'Towards federated learning: An overview of methods and applications,' Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery , vol. 13, no. 2, p. e1486, 2023.
- [44] A. P. Siddaway, A. M. Wood, and L. V. Hedges, 'How to do a systematic review: a best practice guide for conducting and reporting narrative reviews, meta-analyses, and meta-syntheses,' Annual review of psychology , vol. 70, pp. 747-770, 2019.
- [45] Google. (2004) Google scholar. Google. [Online]. Available: http: //scholar.google.com/
- [46] IEEE. (2000) Ieee xplore. IEEE. [Online]. Available: https://ieeexplore. ieee.org/
- [47] U. S. N. L. of Medicine. (1996) Pubmed. United States National Library of Medicine. [Online]. Available: https://pubmed.ncbi.nlm.nih. gov/
- [48] Elsevier. (2004) Scopus. Elsevier. [Online]. Available: https://www. sciencedirect.com/
- [49] Clarivate. (1997) Web of science. Clarivate. [Online]. Available: https://clarivate.com/webofsciencegroup/solutions/web-of-science
- [50] Q. T. Universities. (1994) Qs world university rankings 2024: Top global universities. QS Quacquarelli Symonds Limited. [Online]. Available: https://www.topuniversities.com/ university-subject-rankings/engineering-technology
- [51] X. Wu, J. Pei, X.-H. Han, Y.-W. Chen, J. Yao, Y. Liu, Q. Qian, and Y. Guo, 'Fedel: Federated ensemble learning for non-iid data,' Expert Systems with Applications , vol. 237, p. 121390, 2024.
- [52] J. Liu, T. Che, Y. Zhou, R. Jin, H. Dai, D. Dou, and P. Valduriez, 'Aedfl: efficient asynchronous decentralized federated learning with heterogeneous devices,' in Proceedings of the 2024 SIAM International Conference on Data Mining (SDM) . SIAM, 2024, pp. 833-841.
- [53] B. Li, 'Federated learning with a balanced heterogeneous-yoke and loose restriction,' Internet of Things , vol. 25, p. 101144, 2024.
- [54] Y. Qiao, A. Adhikary, C. Zhang, and C. S. Hong, 'Towards robust federated learning via logits calibration on non-iid data,' arXiv preprint arXiv:2403.02803 , 2024.
- [55] B. Li, M. N. Schmidt, T. S. Alstrøm, and S. U. Stich, 'On the effectiveness of partial variance reduction in federated learning with heterogeneous data,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 3964-3973.
- [56] Y. Dai, Z. Chen, J. Li, S. Heinecke, L. Sun, and R. Xu, 'Tackling data heterogeneity in federated learning with class prototypes,' in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 37, no. 6, 2023, pp. 7314-7322.
- [57] R. Yan, L. Qu, Q. Wei, S.-C. Huang, L. Shen, D. L. Rubin, L. Xing, and Y. Zhou, 'Label-efficient self-supervised federated learning for tackling data heterogeneity in medical imaging,' IEEE Transactions on Medical Imaging , vol. 42, no. 7, pp. 1932-1943, 2023.
- [58] J. Zhang, Z. Li, B. Li, J. Xu, S. Wu, S. Ding, and C. Wu, 'Federated learning with label distribution skew via logits calibration,' in International Conference on Machine Learning . PMLR, 2022, pp. 26 311-26 329.
- [59] Y. He, Y. Chen, X. Yang, H. Yu, Y.-H. Huang, and Y. Gu, 'Learning critically: Selective self-distillation in federated learning on non-iid data,' IEEE Transactions on Big Data , 2022.
- [60] X. Zhang, M. Hong, S. Dhople, W. Yin, and Y. Liu, 'Fedpd: A federated learning framework with adaptivity to non-iid data,' IEEE Transactions on Signal Processing , vol. 69, pp. 6055-6070, 2021.
- [61] Z. Yao, J. Liu, H. Xu, L. Wang, C. Qian, and Y. Liao, 'Ferrari: A personalized federated learning framework for heterogeneous edge clients,' IEEE Transactions on Mobile Computing , 2024.
- [62] Y. Cheriguene, W. Jaafar, H. Yanikomeroglu, and C. A. Kerrache, 'Towards reliable participation in uav-enabled federated edge learning on non-iid data,' IEEE Open Journal of Vehicular Technology , 2023.
- [63] H. Jamali-Rad, M. Abdizadeh, and A. Singh, 'Federated learning with taskonomy for non-iid data,' IEEE transactions on neural networks and learning systems , vol. 34, no. 11, pp. 8719-8730, 2022.

- [64] Y. Huang, L. Chu, Z. Zhou, L. Wang, J. Liu, J. Pei, and Y. Zhang, 'Personalized cross-silo federated learning on non-iid data,' in Proceedings of the AAAI conference on artificial intelligence , vol. 35, no. 9, 2021, pp. 7865-7873.
- [65] Y. Diao, Q. Li, and B. He, 'Exploiting label skews in federated learning with model concatenation,' in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 38, no. 10, 2024, pp. 11 784-11 792.
- [66] Z. Ma, Y. Liu, Y. Miao, G. Xu, X. Liu, J. Ma, and R. H. Deng, 'Flgan: Gan-based unbiased federated learning under non-iid settings,' IEEE Transactions on Knowledge and Data Engineering , 2023.
- [67] J. Shen, N. Cheng, X. Wang, F. Lyu, W. Xu, Z. Liu, K. Aldubaikhy, and X. Shen, 'Ringsfl: An adaptive split federated learning towards taming client heterogeneity,' IEEE Transactions on Mobile Computing , 2023.
- [68] J. Peng, Z. Wu, Q. Ling, and T. Chen, 'Byzantine-robust variancereduced federated learning over distributed non-iid data,' Information Sciences , vol. 616, pp. 367-391, 2022.
- [69] N. M. Jebreel, J. Domingo-Ferrer, D. S´ anchez, and A. Blanco-Justicia, 'Lfighter: Defending against the label-flipping attack in federated learning,' Neural Networks , vol. 170, pp. 111-126, 2024.
- [70] Q. Zhang, Y. Zhu, M. Yang, G. Jin, Y. Zhu, and Q. Chen, 'Crossto-merge training with class balance strategy for learning with noisy labels,' Expert Systems with Applications , vol. 249, p. 123846, 2024.
- [71] S. Jin, Y. Li, X. Chen, R. Li, and Z. Shen, 'Blockchain-based fairnessenhanced federated learning scheme against label flipping attack,' Journal of Information Security and Applications , vol. 77, p. 103580, 2023.
- [72] Z. Li, Z. Lin, J. Shao, Y. Mao, and J. Zhang, 'Fedcir: Clientinvariant representation learning for federated non-iid features,' IEEE Transactions on Mobile Computing , 2024.
- [73] N. Wang, Y. Deng, W. Feng, S. Fan, J. Yin, and S.-K. Ng, 'One-shot sequential federated learning for non-iid data by enhancing local model diversity,' arXiv preprint arXiv:2404.12130 , 2024.
- [74] Y. H. Ezzeldin, S. Yan, C. He, E. Ferrara, and A. S. Avestimehr, 'Fairfed: Enabling group fairness in federated learning,' in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 37, 2023, pp. 7494-7502.
- [75] W. Huang, T. Li, D. Wang, S. Du, J. Zhang, and T. Huang, 'Fairness and accuracy in horizontal federated learning,' Information Sciences , vol. 589, pp. 170-185, 2022.
- [76] N. Onoszko, G. Karlsson, O. Mogren, and E. L. Zec, 'Decentralized federated learning of deep neural networks on non-iid data,' arXiv preprint arXiv:2107.08517 , 2021.
- [77] A. Reisizadeh, F. Farnia, R. Pedarsani, and A. Jadbabaie, 'Robust federated learning: The case of affine distribution shifts,' Advances in Neural Information Processing Systems , vol. 33, pp. 21 554-21 565, 2020.
- [78] M. Badar, S. Sikdar, W. Nejdl, and M. Fisichella, 'Fairtrade: Achieving pareto-optimal trade-offs between balanced accuracy and fairness in federated learning,' in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 38, no. 10, 2024, pp. 10 962-10 970.
- [79] H. Chen, A. Frikha, D. Krompass, J. Gu, and V. Tresp, 'Fraug: Tackling federated learning with non-iid features via representation augmentation,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 4849-4859.
- [80] Z. He, L. Wang, and Z. Cai, 'Clustered federated learning with adaptive local differential privacy on heterogeneous iot data,' IEEE Internet of Things Journal , 2023.
- [81] H. Nguyen, P. Wu, and J. M. Chang, 'Federated learning for distribution skewed data using sample weights,' IEEE Transactions on Artificial Intelligence , 2023.
- [82] Y. Tan, Y. Liu, G. Long, J. Jiang, Q. Lu, and C. Zhang, 'Federated learning on non-iid graphs via structural knowledge sharing,' in Proceedings of the AAAI conference on artificial intelligence , vol. 37, no. 8, 2023, pp. 9953-9961.
- [83] Y. Yan and L. Zhu, 'A simple data augmentation for feature distribution skewed federated learning,' arXiv preprint arXiv:2306.09363 , 2023.
- [84] D. Chiaro, E. Prezioso, M. Ianni, and F. Giampaolo, 'Fl-enhance: A federated learning framework for balancing non-iid data with augmented and shared compressed samples,' Information Fusion , vol. 98, p. 101836, 2023.
- [85] A. Das, T. Castiglia, S. Wang, and S. Patterson, 'Cross-silo federated learning for multi-tier networks with vertical and horizontal data partitioning,' ACM Transactions on Intelligent Systems and Technology (TIST) , vol. 13, no. 6, pp. 1-27, 2022.
- [86] K. Jones, Y. J. Ong, Y. Zhou, and N. Baracaldo, 'Federated xgboost on sample-wise non-iid data,' arXiv preprint arXiv:2209.01340 , 2022.
- [87] T. Feng, D. Bose, T. Zhang, R. Hebbar, A. Ramakrishna, R. Gupta, M. Zhang, S. Avestimehr, and S. Narayanan, 'Fedmultimodal: A benchmark for multimodal federated learning,' in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , 2023, pp. 4035-4045.
- [88] S. Chen and B. Li, 'Towards optimal multi-modal federated learning on non-iid data with hierarchical gradient blending,' in IEEE INFOCOM 2022-IEEE conference on computer communications . IEEE, 2022, pp. 1469-1478.
- [89] K. Borazjani, N. Khosravan, L. Ying, and S. Hosseinalipour, 'Multimodal federated learning for cancer staging over non-iid datasets with unbalanced modalities,' arXiv preprint arXiv:2401.03609 , 2024.
- [90] X. Ouyang, Z. Xie, H. Fu, S. Cheng, L. Pan, N. Ling, G. Xing, J. Zhou, and J. Huang, 'Harmony: Heterogeneous multi-modal federated learning through disentangled model training,' in Proceedings of the 21st Annual International Conference on Mobile Systems, Applications and Services , 2023, pp. 530-543.
- [91] F. Zhou, S. Liu, H. Fujita, X. Hu, Y. Zhang, B. Wang, and K. Wang, 'Fault diagnosis based on federated learning driven by dynamic expansion for model layers of imbalanced client,' Expert Systems with Applications , vol. 238, p. 121982, 2024.
- [92] Y. Cong, J. Qiu, K. Zhang, Z. Fang, C. Gao, S. Su, and Z. Tian, 'Adaffl: Adaptive computing fairness federated learning,' CAAI Transactions on Intelligence Technology , vol. 9, no. 3, pp. 573-584, 2024.
- [93] B. Casella, R. Esposito, A. Sciarappa, C. Cavazzoni, and M. Aldinucci, 'Experimenting with normalization layers in federated learning on noniid scenarios,' IEEE Access , 2024.
- [94] X. Mu, Y. Shen, K. Cheng, X. Geng, J. Fu, T. Zhang, and Z. Zhang, 'Fedproc: Prototypical contrastive federated learning on non-iid data,' Future Generation Computer Systems , vol. 143, pp. 93-104, 2023.
- [95] Y. Wang, Y. Tong, Z. Zhou, R. Zhang, S. J. Pan, L. Fan, and Q. Yang, 'Distribution-regularized federated learning on non-iid data,' in 2023 IEEE 39th International Conference on Data Engineering (ICDE) . IEEE, 2023, pp. 2113-2125.
- [96] M. Morafah, S. Vahidian, W. Wang, and B. Lin, 'Flis: Clustered federated learning via inference similarity for non-iid data distribution,' IEEE Open Journal of the Computer Society , vol. 4, pp. 109-120, 2023.
- [97] Z. Lian, Q. Zeng, W. Wang, T. R. Gadekallu, and C. Su, 'Blockchainbased two-stage federated learning with non-iid data in iomt system,' IEEE Transactions on Computational Social Systems , vol. 10, no. 4, pp. 1701-1710, 2022.
- [98] A. B. de Luca, G. Zhang, X. Chen, and Y. Yu, 'Mitigating data heterogeneity in federated learning with data augmentation,' arXiv preprint arXiv:2206.09979 , 2022.
- [99] Z. Xiong, Z. Cai, D. Takabi, and W. Li, 'Privacy threat and defense for federated learning with non-iid data in aiot,' IEEE Transactions on Industrial Informatics , vol. 18, no. 2, pp. 1310-1321, 2021.
- [100] Y. Mao, Z. Zhao, M. Yang, L. Liang, Y. Liu, W. Ding, T. Lan, and X.P. Zhang, 'Safari: Sparsity-enabled federated learning with limited and unreliable communications,' IEEE Transactions on Mobile Computing , 2023.
- [101] D. Liao, X. Gao, Y. Zhao, and C.-Z. Xu, 'Adaptive channel sparsity for federated learning under system heterogeneity,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 20 432-20 441.
- [102] X. Qiu, J. Fernandez-Marques, P. P. Gusmao, Y. Gao, T. Parcollet, and N. D. Lane, 'Zerofl: Efficient on-device training for federated learning with local sparsity,' arXiv preprint arXiv:2208.02507 , 2022.
- [103] T. Huang, S. Liu, L. Shen, F. He, W. Lin, and D. Tao, 'Achieving personalized federated learning with sparse local models,' arXiv preprint arXiv:2201.11380 , 2022.
- [104] Q. Tong, G. Liang, T. Zhu, and J. Bi, 'Federated nonconvex sparse learning,' arXiv preprint arXiv:2101.00052 , 2020.
- [105] B. Li, P. Song, C. Zhao, and M. Xie, 'Facing spatiotemporal heterogeneity: A unified federated continual learning framework with selfchallenge rehearsal for industrial monitoring tasks,' Knowledge-Based Systems , vol. 289, p. 111491, 2024.
- [106] Y. Tan, C. Chen, W. Zhuang, X. Dong, L. Lyu, and G. Long, 'Is heterogeneity notorious? taming heterogeneity to handle test-time shift in federated learning,' Advances in Neural Information Processing Systems , vol. 36, 2024.
- [107] X. Shen, J. Chen, S. Zhu, and R. Yan, 'A decentralized federated learning-based spatial-temporal model for freight traffic speed forecasting,' Expert Systems with Applications , vol. 238, p. 122302, 2024.
- [108] X. Yang, H. Yu, X. Gao, H. Wang, J. Zhang, and T. Li, 'Federated continual learning via knowledge fusion: A survey,' IEEE Transactions on Knowledge and Data Engineering , 2024.

- [109] F. E. Casado, D. Lema, R. Iglesias, C. V. Regueiro, and S. Barro, 'Ensemble and continual federated learning for classification tasks,' Machine Learning , vol. 112, no. 9, pp. 3413-3453, 2023.
- [110] R. Al-Huthaifi, T. Li, Z. Al-Huda, and C. Li, 'Fedagat: Real-time traffic flow prediction based on federated community and adaptive graph attention network,' Information Sciences , vol. 667, p. 120482, 2024.
- [111] H. Huang, Z. Hu, Y. Wang, Z. Lu, X. Wen, and B. Fu, 'Train a central traffic prediction model using local data: A spatio-temporal network based on federated learning,' Engineering Applications of Artificial Intelligence , vol. 125, p. 106612, 2023.
- [112] S. Chen, G. Long, T. Shen, T. Zhou, and J. Jiang, 'Spatial-temporal prompt learning for federated weather forecasting,' arXiv preprint arXiv:2305.14244 , 2023.
- [113] K. Jin, T. Yin, Z. Chen, Z. Sun, X. Zhang, Y. Liu, and M. Liu, 'Performative federated learning: A solution to model-dependent and heterogeneous distribution shifts,' in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 38, no. 11, 2024, pp. 12 938-12 946.
- [114] M. Yang, H. Qian, X. Wang, Y. Zhou, and H. Zhu, 'Client selection for federated learning with label noise,' IEEE Transactions on Vehicular Technology , vol. 71, no. 2, pp. 2193-2197, 2021.
- [115] I. Jeon, M. Hong, J. Yun, and G. Kim, 'Federated learning via meta-variational dropout,' Advances in Neural Information Processing Systems , vol. 36, 2024.
- [116] I. Wang, P. Nair, and D. Mahajan, 'Fluid: Mitigating stragglers in federated learning using invariant dropout,' Advances in Neural Information Processing Systems , vol. 36, 2024.
- [117] C. Song, D. Saxena, J. Cao, and Y. Zhao, 'Feddistill: Global model distillation for local model de-biasing in non-iid federated learning,' arXiv preprint arXiv:2404.09210 , 2024.
- [118] C. Wang, H. Xia, S. Xu, H. Chi, R. Zhang, and C. Hu, 'Fedbnr: Mitigating federated learning non-iid problem by breaking the skewed task and reconstructing representation,' Future Generation Computer Systems , vol. 153, pp. 1-11, 2024.
- [119] L. Li, D.-c. Zhan, and X.-c. Li, 'Aligning model outputs for class imbalanced non-iid federated learning,' Machine Learning , vol. 113, no. 4, pp. 1861-1884, 2024.
- [120] Z. Lian, J. Cao, Z. Zhu, X. Zhou, and W. Liu, 'Gofl: An accurate and efficient federated learning framework based on gradient optimization in heterogeneous iot systems,' IEEE Internet of Things Journal , 2023.
- [121] H. Zheng, T. Liu, R. Li, and J. Chen, 'Poe: Poisoning enhancement through label smoothing in federated learning,' IEEE Transactions on Circuits and Systems II: Express Briefs , vol. 70, no. 8, pp. 3129-3133, 2023.
- [122] X. You, X. Liu, N. Jiang, J. Cai, and Z. Ying, 'Reschedule gradients: Temporal non-iid resilient federated learning,' IEEE Internet of Things Journal , vol. 10, no. 1, pp. 747-762, 2022.
- [123] D. Yao, W. Pan, Y. Dai, Y. Wan, X. Ding, H. Jin, Z. Xu, and L. Sun, 'Local-global knowledge distillation in heterogeneous federated learning with non-iid data,' arXiv preprint arXiv:2107.00051 , 2021.
- [124] Y. Guo, T. Lin, and X. Tang, 'Towards federated learning on timeevolving heterogeneous data,' arXiv preprint arXiv:2112.13246 , 2021.
- [125] T. Minka, 'Estimating a dirichlet distribution,' 2000.
- [126] J. Lin, 'On the dirichlet distribution,' Department of Mathematics and Statistics, Queens University , vol. 40, 2016.
- [127] Z. Zhou, Y. Li, X. Ren, and S. Yang, 'Towards efficient and stable k-asynchronous federated learning with unbounded stale gradients on non-iid data,' IEEE Transactions on Parallel and Distributed Systems , vol. 33, no. 12, pp. 3291-3305, 2022.
- [128] D. M. Jimenez G., A. Anagnostopoulos, I. Chatzigiannakis, and A. Vitaletti, 'Fedartml: A tool to facilitate the generation of non-iid datasets in a controlled way to support federated learning research,' IEEE Access , vol. 12, pp. 81 004-81 016, 2024.
- [129] Z. Sun, X. Niu, and E. Wei, 'Understanding generalization of federated learning via stability: Heterogeneity matters,' in International Conference on Artificial Intelligence and Statistics . PMLR, 2024, pp. 676-684.
- [130] M. Arafeh, H. Ould-Slimane, H. Otrok, A. Mourad, C. Talhi, and E. Damiani, 'Data independent warmup scheme for non-iid federated learning,' Information Sciences , vol. 623, pp. 342-360, 2023.
- [131] M. Noble, A. Bellet, and A. Dieuleveut, 'Differentially private federated learning on heterogeneous data,' in International Conference on Artificial Intelligence and Statistics . PMLR, 2022, pp. 10 110-10 145.
- [132] K. Hsieh, A. Phanishayee, O. Mutlu, and P. Gibbons, 'The non-iid data quagmire of decentralized machine learning,' in International Conference on Machine Learning . PMLR, 2020, pp. 4387-4398.
- [133] Y. Shi, Z. Liu, Z. Shi, and H. Yu, 'Fairness-aware client selection for federated learning,' in 2023 IEEE International Conference on Multimedia and Expo (ICME) . IEEE, 2023, pp. 324-329.
- [134] R. Ye, Z. Ni, C. Xu, J. Wang, S. Chen, and Y. C. Eldar, 'Fedfm: Anchor-based feature matching for data heterogeneity in federated learning,' IEEE Transactions on Signal Processing , 2023.
- [135] Z. Zhao, C. Feng, W. Hong, J. Jiang, C. Jia, T. Q. Quek, and M. Peng, 'Federated learning with non-iid data in wireless networks,' IEEE Transactions on Wireless communications , vol. 21, no. 3, pp. 19271942, 2021.
- [136] M. Chen, B. Mao, and T. Ma, 'Fedsa: A staleness-aware asynchronous federated learning algorithm with non-iid data,' Future Generation Computer Systems , vol. 120, pp. 1-12, 2021.
- [137] Z. Tang, Z. Hu, S. Shi, Y.-m. Cheung, Y. Jin, Z. Ren, and X. Chu, 'Data resampling for federated learning with non-iid labels,' in Proceedings of the International Workshop on Federated and Transfer Learning for Data Sparsity and Confidentiality in Conjunction with IJCAI, Montreal, Canada . FTLIJCAI, 2021, pp. 21-22.
- [138] H. Wang, Z. Kaplan, D. Niu, and B. Li, 'Optimizing federated learning on non-iid data with reinforcement learning,' in IEEE INFOCOM 2020-IEEE conference on computer communications . IEEE, 2020, pp. 1698-1707.
- [139] H. Zakerinia, S. Talaei, G. Nadiradze, and D. Alistarh, 'Communication-efficient federated learning with data and client heterogeneity,' in International Conference on Artificial Intelligence and Statistics . PMLR, 2024, pp. 3448-3456.
- [140] D. Wang, L. Chen, X. Lu, Y. Wang, Y. Shen, and J. Xu, 'Feddbo: A novel federated learning approach for communication cost and data heterogeneity using dung beetle optimizer,' IEEE Access , vol. 12, pp. 43 396-43 409, 2024.
- [141] H. Yu, C. Wu, H. Yu, X. Wei, S. Liu, and Y. Zhang, 'A federated learning algorithm using parallel-ensemble method on non-iid datasets,' Complex &amp; Intelligent Systems , vol. 9, no. 6, pp. 6891-6903, 2023.
- [142] X. Zhang, Y. Wang, S. Chen, C. Wang, D. Yu, and X. Cheng, 'Robust communication-efficient decentralized learning with heterogeneity,' Journal of Systems Architecture , vol. 141, p. 102900, 2023.
- [143] L. Zhang, Y. Luo, Y. Bai, B. Du, and L.-Y. Duan, 'Federated learning for non-iid data via unified feature learning and optimization objective alignment,' in Proceedings of the IEEE/CVF international conference on computer vision , 2021, pp. 4420-4428.
- [144] A. K. Singh, A. Blanco-Justicia, J. Domingo-Ferrer, D. S´ anchez, and D. Rebollo-Monedero, 'Fair detection of poisoning attacks in federated learning,' in 2020 IEEE 32nd international conference on tools with artificial intelligence (ICTAI) . IEEE, 2020, pp. 224-229.
- [145] K. Wang, Z. Ding, D. K. So, and Z. Ding, 'Age-of-information minimization in federated learning based networks with non-iid dataset,' IEEE Transactions on Wireless Communications , 2024.
- [146] J. Shu, T. Yang, X. Liao, F. Chen, Y. Xiao, K. Yang, and X. Jia, 'Clustered federated multitask learning on non-iid data with enhanced privacy,' IEEE Internet of Things Journal , vol. 10, no. 4, pp. 34533467, 2022.
- [147] H. Zhu, J. Wang, G. Cheng, P. Zhang, and Y. Yan, 'Decoupled federated learning for asr with non-iid data,' arXiv preprint arXiv:2206.09102 , 2022.
- [148] W. Zhang, X. Wang, P. Zhou, W. Wu, and X. Zhang, 'Client selection for federated learning with non-iid data in mobile edge computing,' IEEE Access , vol. 9, pp. 24 462-24 474, 2021.
- [149] C. Yin and Q. Zeng, 'Defending against data poisoning attack in federated learning with non-iid data,' IEEE Transactions on Computational Social Systems , 2023.
- [150] S. Gupta, K. Ahuja, M. Havaei, N. Chatterjee, and Y. Bengio, 'Fl games: A federated learning framework for distribution shifts,' arXiv preprint arXiv:2205.11101 , 2022.
- [151] T. Tang, Z. Han, Z. Cai, S. Yu, X. Zhou, T. Oseni, and S. K. Das, 'Personalized federated graph learning on non-iid electronic health records,' IEEE Transactions on Neural Networks and Learning Systems , 2024.
- [152] E. Rizk, S. Vlaski, and A. H. Sayed, 'Federated learning under importance sampling,' IEEE Transactions on Signal Processing , vol. 70, pp. 5381-5396, 2022.
- [153] K. Selialia, Y. Chandio, and F. M. Anwar, 'Federated learning biases in heterogeneous edge-devices: A case-study,' in Proceedings of the 20th ACM Conference on Embedded Networked Sensor Systems , 2022, pp. 980-986.
- [154] D. J. Beutel, T. Topal, A. Mathur, X. Qiu, J. Fernandez-Marques, Y. Gao, L. Sani, K. H. Li, T. Parcollet, P. P. B. de Gusm˜ ao et al. ,

- 'Flower: A friendly federated learning research framework,' arXiv preprint arXiv:2007.14390 , 2020.
- [155] W. Bao, J. Wu, and J. He, 'Boba: Byzantine-robust federated learning with label skewness,' in International Conference on Artificial Intelligence and Statistics . PMLR, 2024, pp. 892-900.
- [156] H.-Y. Chen and W.-L. Chao, 'Fedbe: Making bayesian model ensemble applicable to federated learning,' arXiv preprint arXiv:2009.01974 , 2020.
- [157] Y. Guo, Q. Tang, Y. Wang, and J. Ren, 'Fdqm: Four-dimensional quantitative measure for statistical heterogeneity in federated learning,' in 2022 IEEE/CIC International Conference on Communications in China (ICCC) . IEEE, 2022, pp. 956-961.
- [158] M. Haller, C. Lenz, R. Nachtigall, F. M. Awayshehl, and S. Alawadi, 'Handling non-iid data in federated learning: An experimental evaluation towards unified metrics,' in 2023 IEEE Intl Conf on Dependable, Autonomic and Secure Computing, Intl Conf on Pervasive Intelligence and Computing, Intl Conf on Cloud and Big Data Computing, Intl Conf on Cyber Science and Technology Congress (DASC/PiCom/CBDCom/CyberSciTech) . IEEE, 2023, pp. 0762-0770.
- [159] H. Zhao, 'Non-iid quantum federated learning with one-shot communication complexity,' Quantum Machine Intelligence , vol. 5, no. 1, p. 3, 2023.
- [160] A. Chen, Y. Fu, Z. Sha, and G. Lu, 'An emd-based adaptive client selection algorithm for federated learning in heterogeneous data scenarios,' Frontiers in Plant Science , vol. 13, p. 908814, 2022.
- [161] T.-M. H. Hsu, H. Qi, and M. Brown, 'Federated visual classification with real-world data distribution,' in Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part X 16 . Springer, 2020, pp. 76-92.
- [162] Q. Tan, S. Wu, and Y. Tao, 'Privacy-enhanced federated learning for non-iid data,' Mathematics , vol. 11, no. 19, p. 4123, 2023.
- [163] U. Ahmed, J. C.-W. Lin, and G. Srivastava, 'Semisupervised federated learning for temporal news hyperpatism detection,' IEEE Transactions on Computational Social Systems , vol. 10, no. 4, pp. 1758-1769, 2023.
- [164] Y. Xu, Y. Li, H. Luo, X. Fan, and X. Liu, 'Fblg: A local graph based approach for handling dual skewed non-iid data in federated learning,' in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24 , K. Larson, Ed. International Joint Conferences on Artificial Intelligence Organization, 8 2024, pp. 5289-5297, main Track. [Online]. Available: https://doi.org/10.24963/ijcai.2024/585
- [165] L. Zhang, G. Gao, and H. Zhang, 'Spatial-temporal federated learning for lifelong person re-identification on distributed edges,' IEEE Transactions on Circuits and Systems for Video Technology , 2023.
- [166] G. Luo, N. Chen, J. He, B. Jin, Z. Zhang, and Y. Li, 'Privacy-preserving clustering federated learning for non-iid data,' Future Generation Computer Systems , vol. 154, pp. 384-395, 2024.
- [167] H. Ling and K. Okada, 'An efficient earth mover's distance algorithm for robust histogram comparison,' IEEE transactions on pattern analysis and machine intelligence , vol. 29, no. 5, pp. 840-853, 2007.
- [168] R. P. Srivastava, 'A new measure of similarity in textual analysis: Vector similarity metric versus cosine similarity metric,' Journal of Emerging Technologies in Accounting , vol. 20, no. 1, pp. 77-90, 2023.
- [169] L. Qu, Y. Zhou, P. P. Liang, Y. Xia, F. Wang, E. Adeli, L. FeiFei, and D. Rubin, 'Rethinking architecture design for tackling data heterogeneity in federated learning,' in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2022, pp. 10 061-10 071.
- [170] S. Zawad, A. Ali, P.-Y. Chen, A. Anwar, Y. Zhou, N. Baracaldo, Y. Tian, and F. Yan, 'Curse or redemption? how data heterogeneity affects the robustness of federated learning,' in Proceedings of the AAAI conference on artificial intelligence , vol. 35, no. 12, 2021, pp. 10 807-10 814.
- [171] C. Park, T. Choi, T. Kim, M. Cho, J. Hong, M. Choi, and J. Choo, 'Fedgeo: Privacy-preserving user next location prediction with federated learning,' in Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems , 2023, pp. 1-10.
- [172] H. Wen, Y. Wu, J. Hu, Z. Wang, H. Duan, and G. Min, 'Communication-efficient federated learning on non-iid data using twostep knowledge distillation,' IEEE Internet of Things Journal , vol. 10, no. 19, pp. 17 307-17 322, 2023.
- [173] J. Ortigosa-Hern´ andez, I. Inza, and J. A. Lozano, 'Measuring the classimbalance extent of multi-class problems,' Pattern Recognition Letters , vol. 98, pp. 32-38, 2017.
- [174] B. Aamer, H. Chergui, M. Benjillali, and C. Verikoukis, 'Entropydriven stochastic federated learning in non-iid 6g edge-ran,' Frontiers in Communications and Networks , vol. 2, p. 739414, 2021.
- [175] H. Yang, M. Fang, and J. Liu, 'Achieving linear speedup with partial worker participation in non-iid federated learning,' arXiv preprint arXiv:2101.11203 , 2021.
- [176] A. Li, J. Sun, B. Wang, L. Duan, S. Li, Y. Chen, and H. Li, 'Lotteryfl: Personalized and communication-efficient federated learning with lottery ticket hypothesis on non-iid datasets,' arXiv preprint arXiv:2008.03371 , 2020.
- [177] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, 'Federated optimization in heterogeneous networks,' Proceedings of Machine learning and systems , vol. 2, pp. 429-450, 2020.
- [178] S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, 'Scaffold: Stochastic controlled averaging for federated learning,' in International conference on machine learning . PMLR, 2020, pp. 5132-5143.
- [179] E. Seo, D. Niyato, and E. Elmroth, 'Resource-efficient federated learning with non-iid data: An auction theoretic approach,' IEEE Internet of Things Journal , vol. 9, no. 24, pp. 25 506-25 524, 2022.
- [180] J. Shao, Y. Sun, S. Li, and J. Zhang, 'Dres-fl: Dropout-resilient secure federated learning for non-iid clients via secret data sharing,' Advances in Neural Information Processing Systems , vol. 35, pp. 10 533-10 545, 2022.
- [181] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, 'Tackling the objective inconsistency problem in heterogeneous federated optimization,' Advances in neural information processing systems , vol. 33, pp. 7611-7623, 2020.
- [182] Y. L. Tun, M. N. Nguyen, C. M. Thwal, J. Choi, and C. S. Hong, 'Contrastive encoder pre-training-based clustered federated learning for heterogeneous data,' Neural Networks , vol. 165, pp. 689-704, 2023.
- [183] H. Lee and D. Seo, 'Fedlc: Optimizing federated learning in noniid data via label-wise clustering,' IEEE Access , vol. 11, pp. 42 08242 095, 2023.
- [184] H. Zou, Y. Zhang, X. Que, Y. Liang, and J. Crowcroft, 'Efficient federated learning under non-iid conditions with attackers,' in Proceedings of the 1st ACM Workshop on Data Privacy and Federated Learning Technologies for Mobile Edge Network , 2022, pp. 13-18.
- [185] Q. Li, B. He, and D. Song, 'Model-contrastive federated learning,' in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2021, pp. 10 713-10 722.
- [186] Z. Wang, J. Xiao, L. Wang, and J. Yao, 'A novel federated learning approach with knowledge transfer for credit scoring,' Decision Support Systems , vol. 177, p. 114084, 2024.
- [187] H. A. Madni, R. M. Umer, and G. L. Foresti, 'Robust federated learning for heterogeneous model and data.' International Journal of Neural Systems , vol. 34, no. 4, pp. 2 450 019-2 450 019, 2024.
- [188] K. Chen, X. Zhang, X. Zhou, B. Mi, Y. Xiao, L. Zhou, Z. Wu, L. Wu, and X. Wang, 'Privacy preserving federated learning for full heterogeneity,' ISA transactions , vol. 141, pp. 73-83, 2023.
- [189] K. Zhang, Y. Dai, H. Wang, E. Xing, X. Chen, and L. Sun, 'Memoryadaptive depth-wise heterogenous federated learning,' arXiv preprint arXiv:2303.04887 , 2023.
- [190] Y. He, Y. Chen, X. Yang, Y. Zhang, and B. Zeng, 'Class-wise adaptive self distillation for heterogeneous federated learning,' in Proceedings of the 36th AAAI Conference on Artificial Intelligence, Virtual , vol. 22, 2022.
- [191] F. Zhang, Y. Zhang, S. Ji, and Z. Han, 'Secure and decentralized federated learning framework with non-iid data based on blockchain,' Heliyon , vol. 10, no. 5, 2024.
- [192] W. Guo, Z. Yao, Y. Liu, L. Zhang, L. Li, T. Li, and B. Wu, 'A new federated learning model for host intrusion detection system under noniid data,' in 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC) . IEEE, 2023, pp. 494-500.
- [193] Z. Zhao, J. Wang, W. Hong, T. Q. Quek, Z. Ding, and M. Peng, 'Ensemble federated learning with non-iid data in wireless networks,' IEEE Transactions on Wireless Communications , 2023.
- [194] Z. Li, J. Shao, Y. Mao, J. H. Wang, and J. Zhang, 'Federated learning with gan-based data synthesis for non-iid clients,' in International Workshop on Trustworthy Federated Learning . Springer, 2022, pp. 17-32.
- [195] Y. J. Cho, J. Wang, and G. Joshi, 'Towards understanding biased client selection in federated learning,' in International Conference on Artificial Intelligence and Statistics . PMLR, 2022, pp. 10 351-10 375.
- [196] J. Zhang, S. Guo, Z. Qu, D. Zeng, Y. Zhan, Q. Liu, and R. Akerkar, 'Adaptive federated learning on non-iid data with resource constraint,' IEEE Transactions on Computers , vol. 71, no. 7, pp. 1655-1667, 2021.
- [197] Y. J. Cho, J. Wang, and G. Joshi, 'Client selection in federated learning: Convergence analysis and power-of-choice selection strategies,' arXiv preprint arXiv:2010.01243 , 2020.