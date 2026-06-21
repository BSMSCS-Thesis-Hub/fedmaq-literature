<!-- image -->

Article

## FedDT: A Communication-Efficient Federated Learning via Knowledge Distillation and Ternary Compression

<!-- image -->

Zixiao He 1,2 , Gengming Zhu 2 , Shaobo Zhang 1,2, * , Entao Luo 3 and Yijiang Zhao 2

- 1 Sanya Institute, Hunan University of Science and Technology, Sanya 572024, China; 22010502017@mail.hnust.edu.cn
- 2 School of Computer Science and Engineering, Hunan University of Science and Technology, Xiangtan 411201, China; zgm@hnust.edu.cn (G.Z.); zhaoyj@hnust.edu.cn (Y.Z.)
- 3 School of Electronics and Information Engineering, Hunan University of Scienceand Engineering, Yongzhou 425199, China; entaoluo@huse.edu.cn
* Correspondence: shaobozhang@hnust.edu.cn

Abstract: Federated learning (FL) enables privacy-preserving collaborative training by iteratively aggregating locally trained model parameters on a central server while keeping raw data decentralized. However, FL faces critical challenges arising from data heterogeneity, model heterogeneity, and excessive communication costs. To address these issues, we propose a communication-efficient federated learning via knowledge distillation and ternary compression framework (FedDT). First, to mitigate the negative impact of data heterogeneity, we pre-train personalized heterogeneous teacher models for each client and employ knowledge distillation to transfer knowledge from teachers to student models, enhancing convergence speed and generalization capability. Second, to resolve model heterogeneity, we utilize the server-initialized global model as a shared student model across clients, where homogeneous student models mask local architectural variations to align feature representations. Finally, to reduce communication overhead, we introduce a two-level compression strategy that quantizes the distilled student model into ternary weight networks layer by layer, substantially decreasing parameter size. Comprehensive evaluations on both MNIST and Cifar10 datasets confirm that FedDT attains 7.85% higher model accuracy and reduces communication overhead by an average of 78% compared to baseline methods. This approach provides a lightweight solution for FL systems, significantly lowering communication costs while maintaining superior performance.

Keywords: federated learning; communication efficiency; data heterogeneity; knowledge distillation; ternary quantization

## 1. Introduction

In recent years, mobile devices have emerged as the primary computational infrastructure utilized by billions of global users, and the scale of mobile devices is expected to exceed tens of billions in the coming years. In this context, smartphones and wearable devices continue to generate massive amounts of data [1,2], The continuous expansion and dynamic variability of data landscapes empower AI systems to achieve rapid performance gains. Traditional machine learning transfers user data to a centralized server to train models centrally. Still, much of the data are inherently privacy-sensitive, leading to a serious risk of privacy leakage in this process [3]. The sharing of privacy-sensitive data [4,5] across different clients/platforms is increasingly restricted by data protection regulations, including the GDPR, in response to mounting privacy concerns [6]. Achieving a balance between open data ecosystems and individual privacy rights requires innovative approaches,

<!-- image -->

Academic Editor: Hung-Yu Chien

Received: 7 May 2025 Revised: 27 May 2025 Accepted: 27 May 2025

Published: 28 May 2025

Citation: He, Z.; Zhu, G.; Zhang, S.; Luo, E.; Zhao, Y. FedDT: A Communication-Efficient Federated Learning via Knowledge Distillation and Ternary Compression. Electronics 2025 , 14 , 2183. https://doi.org/ 10.3390/electronics14112183

Copyright: ©2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/).

<!-- image -->

The concept of federated learning (FL) was first put forward by McMahan et al. [7] and has become an effective solution to resolve this conflict and has been widely used in various fields. Among them, FedAvg [8] is the most representative federated averaging algorithm, whose model performance in Google keyboard prediction represents a mere 1.5% deviation from the centralized training baseline. The successful deployment of FedAvg in production applications proves the feasibility of FL in striking a trade-off between privacy preservation and model efficacy. Although FL demonstrates strong capabilities in safeguarding data privacy, it still faces many problems.

The challenge of data heterogeneity, where data are not independently and identically distributed (IID), continues to hinder the performance of FL systems, particularly in realworld applications with diverse data sources. Reference [9] refers to heterogeneity in data distributions across client devices, which usually stems from the fact that the data on each device correspond to a specific user, geographic location, or time window. For example, FedAvg, a simple weighted average approach, needs to be deployed in environments where the data are independently homogeneous and the model is isomorphic. Still, in practice, data are not independently homogeneous across clients. Karimireddy et al. [10] demonstrated that client-side drift occurs when the data are heterogeneous, leading to poor performance, and introduced an algorithmic solution to address client-side drift in local updates through the utilization of control variables to adjust for client-specific variations during the update procedure, which increases the communication cost between devices. Liao et al. [11] proposed a federated uniform representation enhancement framework, which aims to achieve uniform representation generation for federated unsupervised learning on non-independent and/or non-identically distributed (non-IID) data. The clientside flexible uniform regularizer avoids representation collapse by uniformly dispersing samples, and the server-side efficient uniform aggregator promotes global representation consistency by constraining the uniformity of client-side model updates. The above methods mitigate the drift of the model between global and local levels to some extent; nevertheless, this approach is insufficient to completely eradicate the adverse effects of data heterogeneity on convergence behavior and model accuracy. Hence, addressing data heterogeneity is imperative for improving the effectiveness of FL systems.

The presence of model heterogeneity in federated learning leads to barriers to knowledge transfer between participants. In data-heterogeneous federated learning, client data often have different features and distributions [12], different hardware capabilities [13], or different tasks [14], and the suitable model structure and parameter settings will also be different. However, the traditional federated learning algorithm, the FedAvg global model, is aggregated from the average weights of local models, which cannot meet the demand for customized models in various scenarios and tasks, so each client likes to design its local model independently. For example, Makhija et al. [15] proposed Federated Heterogeneous Neural Networks, which allow each client to build personalized models. In real-world scenarios, heterogeneous models [16] do not match in the parameter space during aggregation, and the global gradient drifts, leading to difficulties in transferring knowledge between clients. Li et al. [17] proposed a decentralized federated learning framework leveraging knowledge distillation to tackle the challenge of knowledge transfer on the client side, allowing the application of federated learning across independently designed models; however, this approach relies on a public dataset and lacks a continuous global model update mechanism, restricting its scalability for new participants whose data characteristics may deviate from existing models and potentially degrade overall performance. To address the issue of model heterogeneity, Diao et al. [18] introduced a parameter subset selection strategy based on device parameters to mitigate parameter space mismatch, although this method may cause weight imbalance due to insufficient training of unshared model com- ponents in limited data subsets. These limitations highlight the critical need to overcome the constraints of traditional aggregation paradigms and knowledge transfer mechanisms to develop more efficient and robust heterogeneous federated learning frameworks that can accommodate diverse model architectures and data distributions while maintaining computational efficiency across resource-constrained devices.

The substantial communication burden in federated learning frameworks remains a critical issue, leads to reduced efficiency in model training, and becomes a major bottleneck for system scaling. Since federated learning requires frequent exchanges of model parameters [19], when deploying large-scale pre-trained models, the high communication overhead poses a huge pressure on bandwidth-constrained clients, which directly restricts the practical application of large-scale models in federated systems [20,21]. To reduce the communication overhead, current research focuses on two main directions: gradient compression and co-distillation. Gradient compression [22] reduces the amount of transmitted data, but is prone to significant performance degradation at high compression ratios and may weaken the model's ability to handle data heterogeneity [23]. The co-distillation paradigm [24] reduces communication overhead by sharing local model predictions rather than transmitting model parameters, and the approach is adopted when the local model's architecture exceeds the representational capacity of the public dataset. However, highly privacy-sensitive data [25] cannot be shared and exchanged in real-world scenarios. Itahara et al. [26] proposed a semi-supervised distillation that reduces communication overhead but still lacks scalability in data-heterogeneous scenarios. The FedKD framework proposed by Wu et al. [27] removes reliance on public data through bidirectional knowledge migration but its performance is limited by the quality of teachers' models. Therefore, balancing communication overhead and model performance remains a key challenge for federated learning.

In summarize, there are still three major problems with federated learning, although it protects data privacy: (1) data heterogeneity leads to significant model performance degradation; (2) model heterogeneity leads to knowledge transfer obstacles; (3) high communication overhead reduces training efficiency. To address these problems, this paper proposes communication-efficient federated learning via knowledge distillation and ternary compression, named FedDT. This method applies the dynamic combination of knowledge distillation and ternary quantization to federated learning, and the client adopts multiple rounds of local update strategy to compress the model simultaneously in two layers. Firstly, an adaptive knowledge distillation mechanism is employed at the client side to transfer knowledge from the teacher model to a lightweight student model. Secondly, the distilled student model undergoes per-layer quantization and is further trained into a ternary-weight network, effectively reducing the parameter count through structured sparsity induction. Finally, the compressed ternary model is aggregated on the server after multiple rounds of local refinement. This approach not only preserves model performance through iterative optimization but also achieves substantial parameter reduction via progressive compression, demonstrating particular advantages in large-scale distributed learning environments where data heterogeneity poses significant challenges.

This study makes the following key contributions:

(1) Personalized Federated Distillation Framework: We introduce a novel algorithm that enables clients to train customized teacher models tailored to their local data distributions. This personalized approach effectively alleviates the adverse effects of data heterogeneity, which is a critical challenge in federated learning, thereby enhancing both model performance and generalization capability.

(2) This paper addresses the challenge of knowledge transfer across heterogeneous local models by employing a student model as a unified intermediary. Specifically, the global model, initialized by the server, functions as a unified student model shared among all participating clients. During local updates, each client performs knowledge distillation from its personalized teacher model into the homogeneous student model, thereby eliminating cross-client knowledge transfer barriers.

(3) This paper proposes a two-level compression strategy that combines knowledge distillation with ternary quantization to reduce model parameters. In federated distillation, communication costs are fundamentally governed by the magnitude of the student model being transmitted. By leveraging ternary quantization, the continuous weight parameters of the student model are mapped to a discrete space, significantly reducing communication costs while maintaining model performance.

(4) The experimental evaluation indicates that FedDT surpasses three baseline methods by 7.85% in model accuracy under both IID and non-IID conditions on the MNIST and Cifar10 datasets, while simultaneously reducing communication overhead by an average of 78%.

This paper adopts a structured approach to present its contributions as follows: The next section conducts a comprehensive review of prevailing research in the federated learning domain, providing critical context for our work. Section 3 introduces the preliminary definitions and the system architecture of FedDT. Section 4 presents the detailed implementation of the two modules within the FedDT framework, along with the overall implementation of the proposed method. Section 5 conducts comparative experiments between FedDT and three baseline methods, analyzing both model performance and communication overhead. Complementary ablation analyses are further performed to verify the individual contributions of each module. Finally, Section 6 summarizes the key findings and outlines promising avenues for future research.

## 2. Related Work

This section presents an in-depth analysis of existing research on federated learning, focusing on challenges related to data heterogeneity, model heterogeneity, and communication overhead.

## 2.1. Data Heterogeneous Federated Learning

Federated learning preserves data confidentiality through localized data storage and processing. The uploaded local parameters are systematically consolidated to derive a unified global model, and this aggregation mechanism may lead to significant deviations between the actual trained global model and the theoretically optimal model [28]. In real-world application scenarios, there are category imbalances both at the global level and in the local dataset of a single client due to industry, geography, culture, and other factors [29,30] that the client is affected by. Unbalanced data bring a double challenge [31]: Initially, the model exhibits a predisposition toward the majority class, and the performance of different classes will be different with an unbalanced number of samples. Second, the lack of minority class data makes it more difficult to learn the features of these classes. Li et al. [32] proposed a solution to FL heterogeneity by modifying local objectives with proximal regularization, improving algorithm stability across heterogeneous clients. Huang et al. [33] developed a personalized FL framework that promotes collaboration between clients with comparable data characteristics. Ruan et al. [34] proposed a federated learning aggregation method that allows more flexible devices to participate in the convergence. Zhang et al. [35] proposed an algorithm to select models with higher frequency. Wang et al. [36] proposed a visual analysis tool, HetVis, to facilitate customers to solve the heterogeneity problem and identified statistical heterogeneity by calculating the predictive behaviors of the global FL model versus the independent models trained using local data.

To solve the client model drift problem caused by statistical heterogeneity among different IoT devices, Zhou et al. [37] proposed a federated learning framework incorporating a global-local knowledge fusion mechanism (FedKF). The core concept of FedKF involves the server returning global knowledge for integration with local knowledge during each training iteration, to regularize the local model into a globally optimal one. Casey et al. [38] proposed a collaborative transfer learning (CTL) framework that utilizes representative datasets and adaptive distillation weights to facilitate efficient and privacy-preserving collaboration. Li et al. [39] proposed MOON, which addresses data heterogeneity by constraining the update direction of local models through a model-contrastive loss that anchors on the feature representations of the global model, while reducing communication overhead via a lightweight global architecture. However, MOON does not resolve clientside model heterogeneity and demonstrates sensitivity to hyperparameter configurations, potentially limiting its applicability in practical scenarios.

The above federated learning solutions can solve the vertical federated learning heterogeneity problem but with an overall level of accuracy and loss. For example, the HetVis algorithm for sample analysis is more complex, while the FedKF algorithm is moderately complex, but with less performance improvement. More efficient and robust schemes need to be investigated in the future for the data heterogeneity problem of FL.

## 2.2. Model Heterogeneous Federated Learning

In practical federated learning settings, the client data's non-IID and diverse distribution precludes the effectiveness of traditional approaches, necessitating independent local model design by each participant to accommodate unique data characteristics [40]. This idea is more adaptable in non-IID scenarios and has gradually become an important direction to solve the problem of data heterogeneity.

Knowledge distillation (KD), first proposed by Hinton et al. [41], is a knowledge transfer method that takes advantage of the rich representations of features embedded in the probability distributions generated by the softmax function to transfer knowledge from a teacher model to a more compact student model, allowing the student to achieve a performance comparable to the teacher. Due to the network architecture of the student model, KD is well suited for deployment on edge devices. Jeong et al. [42] proposed a novel distributed model training framework called the federated distillation algorithm, which pioneered the collaborative training of heterogeneous models across devices, to enhance the personalization capability of the models. Itahara et al. [26] proved the inevitability of the convergence of the process at the mathematical level, providing theoretical support for the regularized constraint term for distributed distillation. The current distillation-based federated learning methods present two major technical trends: first, a lightweight model architecture, typically represented by FedGTK [43] through the periodic knowledge return mechanism, while maintaining the computational independence of the edge device to continuously optimize the central model; second, integration learning fusion, such as FedBoost [44], which adopts an incremental integration framework through the distribution of the basic model components to achieve distributed integrated learning, taking into account the communication efficiency and model performance. Additionally, Tan et al. [45] proposed FedProto, which addresses data and model heterogeneity by exchanging and aggregating class prototypes (rather than gradients) between clients and the server. However, its limitations primarily stem from the insufficient expressiveness of prototypes to capture complex class features and reduced effectiveness in highly heterogeneous or fine-grained classification tasks.

Knowledge distillation techniques show significant bandwidth optimization advantages in federated learning, especially in coping with device heterogeneity and data distri- bution bias, which improves the personalization capability of the model. However, as the number of clients and the dimensions of the data continue to increase, the personalization strategy is prone to become a bottleneck in terms of communication and computational resources. How to trade off efficiency and accuracy among knowledge distillation methods will be a long-term research direction.

## 2.3. Efficient Federal Learning

The client and server need to exchange data, such as model parameters, frequently during the federal learning process. In the present day, the model scale is expanding and the number of parameters is increasing, which is commonly used in computer vision, and has reached 11.69 million. Therefore, model parameters become the main content of federated learning communication. Considering that some of the information in deep learning models is redundant and the non-absolute dependence of most models on numerical accuracy, quantization compression techniques can achieve the efficient use of computational resources with controlled loss of accuracy. However, this means of optimization also poses some challenges, such as the increased complexity of the training process and potential degradation of model accuracy. To address the problem that quantization methods may lead to excessive sparsity, researchers have proposed a ternary quantization strategy to reduce the quantization error by introducing a zero value as the third quantization level. Zhu's team [46] proposed the trained ternary quantization method (TTQ), which implements asymmetric quantization by introducing two trainable scaling coefficients { -Wn , 0, Wp }, thereby improving the expressive power of the network. Ternary quantization simplifies computation and storage by mapping weights or activation values to three discrete values. This approach is particularly suitable for deep learning models and can significantly reduce model complexity while preserving as much performance as possible. In federated learning scenarios, quantization compression techniques present two major innovative paths. Fixed quantization-based methods optimize the quantization error through matrix transformation, such as AdaGQ [47]. Dynamically adjusting the quantization bit width and combining the differential allocation of communication capacity can increase the convergence speed by 35%, and the budget-driven EDEN communication strategy [48] can reduce energy consumption by 28%. These methods significantly improve the efficiency of model convergence in heterogeneous environments.

Quantization technology not only reduces the model storage requirement by 50-80% but also reduces the transmission delay by 40-60% by reducing the amount of data per round of communication by 3-5 times, providing key technical support for edge computing scenarios. However, the quantization technique is a lossy compression technique, which restricts the high-precision representation of the model and is not suitable for some applications or tasks that require high precision, so it is necessary to further explore and improve the quantization technique in federated learning to adapt to more complex and wider application scenarios.

In summary, existing federated learning methods suffer from critical limitations including model performance degradation due to data heterogeneity, knowledge transfer barriers caused by model heterogeneity, and excessive communication overhead. These challenges require the development of novel solutions. As shown in Table 1, where × indicates negative impacts, - indicates no improvement, and ✓ indicates positive outcomes, the proposed framework demonstrates comprehensive advantages in addressing data heterogeneity, model heterogeneity, and communication efficiency compared to alternative approaches.

Table 1. Comparison of representative methods in terms of data heterogeneity, model heterogeneity, and communication overhead.

| Method          | Data- Heterogeneous   | Model- Heterogeneous   | Communication Overhead   |
|-----------------|-----------------------|------------------------|--------------------------|
| FedKD [27]      | ✓                     | ✓                      | ✓                        |
| FedProx [32]    | ✓                     | -                      | ×                        |
| FedAMP [33]     | ✓                     | -                      | -                        |
| CSFedAvg [35]   | ✓                     | -                      | -                        |
| HetVis [36]     | ✓                     | -                      | ×                        |
| FedKF [37]      | ✓                     | -                      | ×                        |
| MOON[39]        | ✓                     | ×                      | ✓                        |
| CTL [38]        | ✓                     | -                      | ✓                        |
| Feddistill [42] | ✓                     | ✓                      | ✓                        |
| FedGTK [43]     | -                     | ✓                      | -                        |
| FedBoost [44]   | -                     | ✓                      | ✓                        |
| AdaGQ [47]      | -                     | -                      | ✓                        |
| FedProto [45]   | ✓                     | ✓                      | ✓                        |
| FedDT (Ours)    | ✓                     | ✓                      | ✓                        |

## 3. System Architecture

## 3.1. Preliminary Definitions

Definition 1 (Federated Learning) . Let N be the number of participating clients, denoted as F = { F 1 , F 2, . . . , FN } , where each client F i possesses a private local dataset, D i . In conventional centralized learning, all data are aggregated as D = D 1 ∪ D 2 ∪ · · · ∪ DN, and a global model, MSUM, is trained on the combined dataset. In contrast, federated learning enables the training of the collaborative model of MFL without requiring any client F i to disclose D i to others. Let V SUM and VFL denote the performance metrics of the centralized and federated models respectively, under the condition that

$$
| V _ { F e d } - V _ { S U M } | < \delta
$$

Then, the federated learning algorithm is said to have a loss of δ in accuracy.

Definition 2 (Non-IID Data) . Let the local data distribution of user i be ( x , y ) ∼ Pi ( x , y ) and that of user j be ( x , y ) ∼ Pj ( x , y ) , where there is a statistical discrepancy between them. Each user's data distribution possesses the property of decomposition into the product of a feature distribution, P ( x ) , and a label-conditional distribution, P ( x | y ) :

̸

$$
P _ { i } ( y _ { i } \ | \ x _ { i } ) \cdot P ( x _ { i } ) \neq P _ { j } ( y _ { j } \ | \ x _ { j } ) \cdot P ( x _ { j } )
$$

The situation of non-IID data labels refers to the case where the conditional probabilities P i ( y | x ) and Pj ( y | x ) among different users are the same, but their marginal probabilities P i ( x ) and Pj ( x ) are different. This paper focuses on federated learning algorithms under label-skewed non-IID data conditions.

Definition 3 (Softmax Function) . Knowledge distillation employs a generalized softmax function with a temperature parameter:

$$
q _ { j } ( T , w ) = \frac { \exp ( z _ { j } ( w ) / T ) } { \sum _ { j } \exp ( z _ { j } ( w ) / T ) }
$$

where T is the hyper-parameterized average temperature, which is used to control the smoothness of knowledge loss. The temperature hyperparameter is utilized to regulate the teacher model's output, avoiding overly peaked probability distributions. As the temperature value rises, the output probability distribution becomes increasingly smooth. Here, z i represents the logarithmic output of the teacher model, with i indicating the index of the ith class, j denoting the total number of classes, and pij indicating the predicted probability for the ith class. The term q j ( T , w ) corresponds to the class probability prediction for the ith class obtained from the softmax outputs of both teacher and student models at the temperature T. This method enhances the student model's performance by incorporating the teacher model's predictions as informative soft labels, which function as effective regularization terms during training.

Definition 4 (Cross-Entropy Loss Function) . In machine learning, the cross-entropy loss function is employed to quantify the mismatch between a model's predicted probability distribution and the actual probability distribution. Its mathematical expression is

$$
C E ( a , b ) = - \sum _ { i } a _ { i } \log ( b _ { i } )
$$

where ai is the i-th component of the true label (0 or 1); bi is the predicted probability of the model (softmax output) for the i-th class. Here, a i represents the i-th component of the ground truth label, taking a binary value of either 0 or 1, while b i denotes the predicted probability of the model for the i-th class, derived from the softmax output.

Definition 5 (Kullback-Leibler Divergence) . The Kullback-Leibler (KL) divergence, a measure of dissimilarity between two discrete probability distributions, P and Q, is mathematically defined as

$$
D _ { K L } ( P \left \| \, Q \right ) = \sum _ { i } P ( i ) \log \frac { P ( i ) } { Q ( i ) }
$$

Here, P ( i ) denotes the probability of the i-th element in distribution P, while Q ( i ) represents the corresponding probability in distribution Q. A lower KL divergence value reflects a closer alignment between the two probability distributions. The KL divergence possesses important properties including non-negativity, asymmetry, and the non-satisfaction of the triangle inequality. Specifically, it equals zero when P and Q are identical and approaches infinity when they are completely dissimilar. The KL divergence can be interpreted as measuring either information gain (the amount of information obtained when updating from Q to P) or information loss (the amount of information lost when approximating P with Q). As an asymmetric measure, it finds practical applications in comparing similarity/dissimilarity between texts in natural language processing or between images in computer vision tasks.

## 3.2. System Model of FedDT

The problem of model heterogeneity and communication inefficiency in FL is addressed in this work for data exhibiting non-independent and non-identical distributions; this paper proposes communication-efficient FL via knowledge distillation and ternary compression, named FedDT. A communication-efficient personalized FL model for non-IID scenarios is constructed. The system architecture of FedDT consists of N clients and a server, as shown in Figure 1.

Figure 1. The overall framework of the proposed method.

<!-- image -->

- (a) Clients : Each client locally stores its private data at U = { D 1 , D 2, . . . , DN } . Client data are securely retained within local storage environments. The i-th client's dataset is designated as Di , with client possessing-defined computational capacities. Each participant stores a localized version of the comprehensive tutor model Ti (parameters: Θ i ) and a localized instance of the efficient student model S with its parameter configuration.
- (b) Server : The center server has the characteristics of low latency, high throughput, and high reliability, is capable of handling a large amount of data, and undertakes important tasks such as coordinating the aggregation of client parameters and model updating. Its main tasks are aggregating quantitative models and sending out updated global models.

The main process of FedDT is as follows.

Global model initialization is performed before FL begins. The central server initializes a random global model. Each client downloads the global model as a local student model and pre-trains a personalized teacher model using a local private dataset.

- (1) Client-side local model distillation training. The client uses locally labeled data to optimize model parameters through knowledge migration using the self-distillation mechanism to perform local teacher model and student model updates.
- (2) Client-side student model quantization. The client compresses the student model using a ternary quantization technique. The student model's weights are normalized and then quantized.
- (3) Client-side uploading of quantization models. The client-side quantitative models participating in the training are uploaded to the server.

- (4) The server aggregates the client-uploaded models to derive a global consensus model by utilizing an aggregator to transform each quantized local model into a continuous representation, subsequently synthesizing these into a unified global model.
- (5) Quantization of global models on the server. The aggregated global model is quantized and compressed.
- (6) Clients download the global model. Each participating training client retrieves the quantized global model to perform local model updates.

The optimization process iterates until both the student and teacher models achieve the convergence criteria, as evidenced by the diminishing gradient updates or the saturation of the validation metric.

## 4. Method

In this section, we present the implementation of FedDT in the knowledge distillation module and the ternary quantization module on the local side and the overall implementation of the FedDT framework.

## 4.1. Knowledge Distillation Module

To address the challenges posed by data and model heterogeneity in FL, this study introduces a novel personalized FL approach. Specifically, each client trains a customized heterogeneous teacher model tailored to its unique data distribution, effectively mitigating the adverse effects of data heterogeneity on model optimization. The server maintains a global model that serves as a unified student model across all clients. Through the knowledge distillation process, clients transfer knowledge from their personalized teacher models to this shared student model, thereby obfuscating local model heterogeneity while preserving global consistency. This mechanism leverages the inherent properties of knowledge distillation, where the student model acts as an intermediary to harmonize diverse local models. The detailed local update mechanism is illustrated in Figure 2.

Figure 2. Knowledge distillation process diagram.

<!-- image -->

The local knowledge distillation loss function is composed of three distinct components: task loss function, distillation loss function, and adaptive hidden loss.

(1) Task loss function: The discrepancy between the model's predictions and actual labels is measured by the cross-entropy loss, serving to refine the model's classification accuracy. When the input sample pair ( xi , yi ) is input, the soft probabilities of the teacher model and the student model after the prediction of the sample xi are denoted as y t i and y s i , respectively. The task loss is denoted as follows.

$$
F _ { ( t , i ) } ^ { t } = C E ( y _ { i } , y _ { i } ^ { t } )
$$

$$
F _ { ( s , i ) } ^ { t } = C E ( y _ { i } , y _ { i } ^ { s } )
$$

(2) Distillation loss function: The training objective combines teacher model soft labels (from its softmax output) with student model hard labels (from its predictions). The distillation loss is computed as the KL divergence between these soft labels, enabling the student to mimic the teacher's intermediate representations and approximate its output distribution.

In federated knowledge distillation, the teacher-student training dynamics directly influence the loss balance. When models converge effectively, the distillation loss takes precedence, effectively curbing overfitting while possibly impairing the student model's ability to predict real labels accurately. In contrast, during early training stages or when data noise is pronounced, inadequate prediction reliability causes the task loss to prevail, obstructing efficient knowledge transfer. To overcome these issues, this paper introduces an adaptive distillation mechanism predicated on soft label quality perception as follows.

$$
F _ { ( t , i ) } ^ { d } = \frac { K L ( y _ { i } ^ { s } , y _ { i } ^ { t } ) } { F _ { ( t , i ) } ^ { t } + F _ { ( s , i ) } ^ { t } }
$$

$$
F _ { ( s , i ) } ^ { d } = \frac { K L ( y _ { i } ^ { t } , y _ { i } ^ { s } ) } { F _ { ( t , i ) } ^ { t } + F _ { ( s , i ) } ^ { t } }
$$

Adaptive intensity control by dynamically adjusting loss weights based on the predicted correctness of the teacher and student models. In the scenario of a high correct rate, the distillation loss weight is reduced and task learning is focused on. In scenarios with low correct rates, distillation loss weights are increased to enhance knowledge transfer. Distillation loss is balanced with task loss by using temperature and weighting methods to facilitate student model training.

(3) Due to the hidden state of the teacher model and the fact that the attention heat map contains key features of the data with contextual dependencies, additional adaptive hidden loss functions are added on top of traditional knowledge distillation techniques. The student model learns a more robust feature extraction capability by matching these representations. The loss formulas F h ( t , i ) and F h ( s , i ) are as follows.

$$
F _ { ( t , i ) } ^ { h } = F _ { ( s , i ) } ^ { h } = \frac { M S E ( H _ { i } ^ { t } , W _ { i } ^ { h } H ^ { s } ) + M S E ( A _ { i } ^ { t } , A ^ { s } ) } { F _ { ( t , i ) } ^ { t } + F _ { ( s , i ) } ^ { t } }
$$

The mean squared error (MSE) is utilized as the key objective function for optimization in this work. Let H t i , A i , ˆ H i , and ˆ A i represent the hidden state and the heat of attention map in the i th local teacher and student, respectively. Additionally, let W h i be a parameterized linear transformation matrix. We design an adaptive scaling method for the hidden loss function, dynamically adjusted by the teacher-student prediction accuracy. In summary, the unified loss function for local updates of the teacher and student models for each client is formulated as follows:

$$
F _ { ( t , i ) } = F _ { ( t , i ) } ^ { d } + F _ { ( t , i ) } ^ { h } + F _ { ( t , i ) } ^ { t }
$$

$$
F _ { ( s , i ) } = F _ { ( s , i ) } ^ { d } + F _ { ( s , i ) } ^ { h } + F _ { ( s , i ) } ^ { t }
$$

The comprehensive loss function is constructed by aggregating the distillation loss, task loss, and adaptive hidden loss. By implementing the backpropagation algorithm, the cumulative loss is reduced to enhance the model's learning efficiency, and the gradient gi of the student model on the i th client can be derived from F ( s , i ) through gi = ∂ F ( s , i ) ∂ Θ s , where Θ s represents the parameter set of the student model. The local teacher model for each client is updated by the local gradient obtained from the loss function F ( t , i ) . FL, using the knowledge distillation model compression method, has a communication load that depends on the size of the output student model. By capitalizing on the properties of knowledge distillation, the method allows the student model to conceal the heterogeneity of the local model, which helps to mitigate the device heterogeneity and statistical heterogeneity of the data.

## 4.2. Ternary Quantization Module

To reduce the communication overhead, a two-layer compression strategy is used, and this section is the ternary quantization phase of the local model update, and the local ternary quantization process is shown in Figure 3. For the local client, the distilled student model is trained twice using labeled data, and the student model weights are mapped layer by layer to three discrete values (usually -1, 0, and +1) during the training process to simplify computation and storage. By quantizing the model into a ternary weight network during the training process, this method significantly reduces the model complexity while preserving its performance as much as possible, which is particularly suitable for deep learning.

Figure 3. Ternary quantization process diagram.

<!-- image -->

First, the weights of the student model are subjected to a normalization operation.

$$
\Theta ^ { s } = g ( \Theta )
$$

Θ denotes the full-precision weight matrix of the student model, Θ s represents the normalized weight matrix, and g signifies the normalization function, which normalizes a vector to a certain random range to make the weight distributions of different layers closer, making the subsequent quantization more stable. Based on normalization, the continuous weights of the student model are discretized into three values ( -1, 0, +1) by threshold division, which significantly reduces the storage and computation overhead. Thresholds are determined by generating uniformly distributed random numbers based on the sparsity of the weights used.

$$
\Delta = \frac { T _ { k } } { d ^ { 2 } } \sum _ { i } ^ { d ^ { 2 } } | \Theta _ { i } ^ { s } |
$$

where Tk denotes the parameter set of client k, ∆ signifies the adaptive optimum value, d indicates the number of layers, and Tk is set to 0.7.

$$
\begin{aligned}
T _ { k } = f ( x ) = \begin{cases} 0 . 0 5 + 0 . 0 1 \cdot \text {rand} ( 0 , 1 ) , & \text {if} \, \text {rand} ( 0 , 1 ) > 0 . 5 \\ 0 . 0 5 + 0 . 0 1 \cdot \frac { k } { N } , & \text {if} \, \text {rand} ( 0 , 1 ) \leq 0 . 5 \end{cases}
\end{aligned}
$$

Subsequently, realizing layer-by-layer weight quantization is adopted, in which the quantization accuracy is balanced by adaptive thresholds based on the number of layers and global scaling factors, breaking through the limitations of traditional fixed thresholds.

$$
\text {mask} ( \Theta ^ { s } ) = \varepsilon ( | \Theta ^ { s } | - \Delta )
$$

$$
I ^ { t } = \text {sign} ( \text {mask} \odot \Theta ^ { s } )
$$

$$
\Theta ^ { s } = \omega ^ { q } \times I ^ { t }
$$

In this context, ε denotes a step function, the Hadamard product mask ( Θ s ) applies a thresholding operation that sets elements to 1 when their absolute values surpass a certain threshold, ω q is a layer-specific quantization factor that undergoes layer-wise training in conjunction with the local model's weights, and It represents the quantized ternary weights. Therefore, mask ( Θ s ) can be expressed as the concatenation of the positive indicator matrix Ip and the negative indicator matrix In .

$$
I _ { P } = \{ i \, | \, \Theta _ { i } ^ { s } > \Delta \}
$$

$$
I _ { n } = \{ i \, | \, \Theta _ { i } ^ { s } < - \Delta \}
$$

$$
w _ { p } = \frac { 1 } { | I _ { p } | } \sum _ { i \in I _ { p } } \Theta _ { i }
$$

$$
w _ { n } = - \frac { 1 } { | I _ { n } | } \sum _ { j \in I _ { n } } \Theta _ { j }
$$

Upon completing the quantization of the entire network, the loss function is calculated, and the error is transmitted backward via backpropagation. ω q and the gradient of the potential full-precision model is computed as follows.

$$
\frac { \partial J } { \partial \omega ^ { q } } = \sum _ { i \in I _ { P } } \frac { \partial J } { \partial \Theta _ { i } ^ { t } }
$$

$$
\begin{aligned}
\frac { \partial J } { \partial \omega ^ { q } } = \begin{cases} 1 \cdot \frac { \partial J } { \partial \Theta ^ { t } } , & | \Theta | \leq \Delta \\ \omega ^ { q } \cdot \frac { \partial J } { \partial \Theta ^ { f } } , & \text {otherwise} \end{cases}
\end{aligned}
$$

Reducing upstream and downstream communication overhead by quantizing distilled student models also enhances privacy preservation in FL due to the lower weights making it harder to reverse inference sensitive data.

The local model ternary quantization Algorithm 1 is as follows:

## Algorithm 1: Local model ternary quantization TTQ ( w q , Θ s k ) .

```
Input: post-distillation model Θ s k , quantization factor w q , sample pairs ( xi , yi ) , learning rate η , function l Output: quantitative modeling Θ s 1 for ( xi , yi ) ∈ D do 2 Θ s = g ( Θ s k ) ; 3 mask ( Θ s ) = ϵ ( | Θ s | -∆ ) ; 4 I ∗ = sign ( mask ⊙ Θ s ) ; 5 Θ s r ← w q × I t ; 6 J ← l t ( xi , yi ; Θ t ) ; 7 w q ← w q + η ∂ J ∂ w q ; 8 Θ s ← Θ s r + η ∂ J ∂ Θ q ; 9 end
```

## 4.3. FedDT

The FedDT method uses a multi-round local update strategy, combining two modules, knowledge distillation and ternary quantization. The primary aim is to boost the global model's accuracy via successive training iterations, while performing model compression during training minimizes storage consumption. The client-side local update process of the FedDT method is illustrated in Figure 4.

Figure 4. FL client local update diagram.

<!-- image -->

The FedDT-specific training process is as follows:

In the context of FL, the server first initializes a global model, Θ , with random weights before the training process begins. Clients subsequently download this model from the server and utilize it as their local student model, Θ s . Acknowledging the potential association between client data distributions and model parameters, this study introduces a personalized teacher model, Θ i t , which is pre-trained for each client using labeled private data to align with the specific patterns of its local data distribution.

- (1) Client-Side Local Model Distillation Training. This study employs three distinct loss functions-task loss (Equation (4)), distillation loss (Equation (8)), and hidden loss (Equation (10))-to facilitate reciprocal knowledge transfer between the teacher and student models during training. Dynamic weight allocation is performed throughout the knowledge distillation process, with weights adjusted based on the intensity of either the task loss or distillation loss to enhance student model training. Furthermore, an adaptive hidden loss function is introduced to enable the student model to acquire knowledge from the teacher model's hidden states (Ht) and attention heatmaps (At).

$$
g _ { i } ^ { t } \leftarrow \frac { \partial F _ { ( s , i ) } } { \partial \Theta _ { i } ^ { t } }
$$

$$
g _ { i } ^ { s } \leftarrow \frac { \partial F _ { ( t , i ) } } { \partial \Theta _ { i } ^ { s } }
$$

- (2) Local Model Quantization on the Client Side. To further decrease communication overhead and improve model accuracy, the locally labeled data are employed for retraining the student model. Normalization techniques are applied to the student model's weights during training. On top of normalization, the weights are quantized from full-precision floating-point numbers to a ternary representation ( -1, 0, 1) with reduced bit-width, compressing the student model into an adaptive ternary quantized weight network.

$$
\Theta _ { k , r } ^ { s } = T T Q ( w ^ { q } , \Theta _ { k } ^ { s } )
$$

- (3) Small Model Uploading by the Client. The local quantization model Θ s k , r for the clients participating in this training is uploaded to the server.
- (4) The central server conducts aggregation of miniature models. During each communication cycle, the server converts the uploaded quantized local models into continuous counterparts and fuses them into the global model.

$$
\Theta _ { r } \leftarrow \sum _ { k = 1 } ^ { \lambda N } \frac { | D _ { k } | } { \sum _ { k = 1 } ^ { \lambda N } | D _ { k } | } \Theta _ { k , r } ^ { s }
$$

(5) The server quantizes the global model with the threshold ∆ s = 0.05 × max ( | Θ r | ) and two quantization factors, ω s p = 1 | I s P | ∑ | I s P | i = 1 | Θ i r | and ω s n = 1 | I s n | ∑ | I s n | i = 1 | Θ i r | , as defined in Equations (11) and (12). After completing the aggregation process, the server propagates the quantized global model to the client.

$$
\Theta _ { r } \leftarrow \omega _ { p } \times I _ { p } - \omega _ { n } \times I _ { n }
$$

- (6) Clients obtain the global model. Each participating client in the training phase downloads the quantized global model to perform local model updates. This cycle repeats until the student model and the global model converge.

Algorithm 2 shows the detailed process of knowledge distillation and ternary quantization.

## Algorithm 2: FedDT.

Input: training dataset, iteration rounds T, participation rate r, initial global model parameters Θ s .

Output: Personalized FL model of the client Θ t k , updated global model Θ s r .

- 1 Initialization Initialize all client teacher models Θ s i and global model Θ s , broadcast to all clients.;
- 21 Return updated global model Θ r to clients for next round training.

```
2 for Each Client k ∈ K = { 1, 2, . . . , | N |} do 3 Θ s k ← Θ s r ; 4 F d ( t , i ) = KL ( y s i , y t i ) F t ( t , i ) + F t ( s , i ) ; 5 F d ( s , i ) = KL ( y t i , y s i ) F t ( t , i ) + F t ( s , i ) ; 6 F h ( t , i ) = F h ( s , i ) = MSE ( H t i , W h i H s )+ MSE ( A t i , A s ) F t ( t , i ) + F t ( s , i ) ; 7 F ( t , i ) = F d ( t , i ) + F h ( s , i ) + F t ( t , i ) ; 8 F ( s , i ) = F d ( s , i ) + F h ( s , i ) + F t ( s , i ) ; 9 g t i ← ∂ F ( s , i ) ∂ Θ t i ; 10 g s i ← ∂ F ( t , i ) ∂ Θ s i ; 11 initialize ω q ; 12 Θ s k , r = TTQ ( ω q , Θ s k ) ; 13 end 14 for Server do 15 Θ r ← ∑ λ N k = 1 | Dk | ∑ λ N k = 1 | Dk | Θ s k , r ; 16 ∆ s = 0.05 × max ( | Θ r | ) ; 17 I S p , I S n = { i | Θ i r > ∆ s } , { i | Θ i r < -∆ s } ; 18 ω S p , ω S n = 1 | I S p | ∑ | I S p | i = 1 | Θ ( i ) r | , 1 | I S n | ∑ | I S n | i = 1 | Θ ( i ) r | ] ; 19 Θ r ← ω p × Ip -ω n × In ; 20 end
```

## 5. Theoretical Analysis

In this section, we conduct both unbiasedness and convergence proofs for the local knowledge distillation module and the ternary quantization module within the FedDT framework.

## 5.1. Unbiasedness of FedDT

The unbiasedness proof aims to demonstrate that the output results of the FedDT algorithm, after passing through the knowledge distillation module and the ternary quantization module, are unbiased in the expectation sense. We will separately analyze and prove that the expected outputs of the entire knowledge distillation module and the ternary quantization module are equal to the true unbiased estimates.

Assumption 1. For all k ∈ [ K ] and any E k [ ∥∇ Fk ( w ) ∥ 2 ] ≤ B 2 ∥∇ f ( w ) ∥ 2 , the expectation is taken with respect to the client index k, where w denotes model parameters and B is the heterogeneity bound.

Proposition 1. In the local knowledge distillation process of federated learning, where the student model optimizes its parameters by learning from the teacher model's knowledge, we have the following:

$$
\mathbb { E } ( \Theta ) = \mathbb { E } \left [ F _ { ( s , i ) } \right ]
$$

Proof of Proposition 1. According to Equation (4), the task loss function adopts the crossentropy loss, which is computed based on the true label yi and the model prediction y t i . Furthermore, during the data sampling process, the samples ( xi , yi ) are independently and identically distributed. Let p ( y | xi ) denote the true label distribution for sample xi . Since the model training objective is to minimize the cross-entropy loss, the task loss functions F t ( t , i ) = CE ( yi , y i t ) and F t ( s , i ) = CE ( yi , y i s ) satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The task loss function is unbiased in expectation because the model optimizes towards minimizing the cross-entropy between predictions and true labels.

According to Equation (5), the distillation loss function is based on Kullback-Leibler (KL) divergence, which measures the difference between two probability distributions. During knowledge distillation, the teacher model's soft labels y i t approximate the true data distribution. As the teacher and student models optimize during training, their predictions converge towards the true data distribution. Thus, the distillation loss satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The distillation loss is unbiased in expectation because the soft labels of teacher model guide the student model to learn a distribution close to the true data distribution.

The adaptive hidden loss function F h ( t , i ) = F h ( s , i ) is defined as

<!-- formula-not-decoded -->

where H t i and A t i are the hidden states and attention heatmaps of the teacher model and H i s and A i s are those of the student model. During training, these representations are optimized to approximate the true data features. Thus, the adaptive hidden loss satisfies

$$
\mathbb { E } [ F _ { ( t , i ) } ^ { h } ] = \mathbb { E } \left [ M S E ( H _ { i } ^ { t } , W H _ { i } ^ { s } ) + M S E ( A _ { i } ^ { t } , A _ { i } ^ { s } ) \right ]
$$

The adaptive hidden loss is unbiased in expectation because the hidden states and attention heatmaps are optimized to capture true data features. According to Equations (11) and (12), the total loss function for the knowledge distillation module is

$$
\mathbb { E } [ F _ { ( s , i ) } ] = \mathbb { E } [ F _ { ( s , i ) } ^ { d } ] + \mathbb { E } [ F _ { ( s , i ) } ^ { h } ] + \mathbb { E } [ F _ { ( s , i ) } ^ { t } ]
$$

$$
\mathbb { E } [ F _ { ( t , i ) } ] = \mathbb { E } [ F _ { ( t , i ) } ^ { d } ] + \mathbb { E } [ F _ { ( t , i ) } ^ { h } ] + \mathbb { E } [ F _ { ( t , i ) } ^ { t } ]
$$

During the knowledge distillation process, the student model optimizes its own parameters by learning from the teacher model's knowledge. Therefore, the output of knowledge distillation can be regarded as an unbiased estimator of the input.

To theoretically prove the unbiasedness of the ternary quantization module in FedDT, we introduce the following assumption.

Assumption 2. The elements of the normalized full-precision parameter Θ are uniformly distributed between -1 and 1, that is,

$$
\forall \Theta _ { i } \in \Theta , \Theta _ { i } \sim U ( - 1 , 1 )
$$

Based on Assumption 2, we prove Proposition 2:

Proposition 2. Let Θ be the local scaled network parameters of a certain client in a given federated learning system. If Θ is quantized by the TTQ algorithm, then

$$
\mathbb { E } [ T T Q ( \Theta ) ] = \mathbb { E } ( \Theta )
$$

Proof of Proposition 2. According to Equations (21) and (22), wq is calculated from the elements of the set I = { i | θ i ≥ ∆ } , where ∆ is a fixed number after parameter generation under Assumption 1. Therefore, the elements indexed by I follow a uniform distribution between ∆ and 1, that is,

<!-- formula-not-decoded -->

So, the probability density function f of Θ i ( i ∈ Ip ) can be expressed as f ( x ) = ( 1/ ( 1 -∆ )) . According to Proposition 2, we can obtain

<!-- formula-not-decoded -->

Let u be a random variable following the distribution in Equation (39), and I represents the number of elements in the set I. We know that

<!-- formula-not-decoded -->

It can be further transformed into

<!-- formula-not-decoded -->

Then, we calculate E [ mask ( Θ )] , and its value is

<!-- formula-not-decoded -->

Under Assumption 2, we have

<!-- formula-not-decoded -->

From this, we can immediately obtain the following:

<!-- formula-not-decoded -->

Therefore, the output of the ternary quantizer can be considered an unbiased estimate of the input. When the weights are uniformly distributed, we can ensure the unbiasedness of FedDT in the federated learning system.

## 5.2. Convergence Analysis

We have proven that the unbiasedness of FedDT holds. This section conducts a convergence analysis of FedDT.

Proposition 3. Let J 1 , . . . , JN be L-smooth and µ -strongly convex functions.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ξ k , r are the mini-batch data points uniformly randomly sampled from the data of the k-th client.

In addition, we assume that the smallest eigenvalue of the client loss function ∇ 2 f k ( w ) is uniformly bounded below by a constant, λ min ∈ R . For all k ∈ [ K ] and any w ∈ R d , E k [ ∥∇ Fk ( w ) ∥ 2 ] ≤ B 2 ∥∇ f ( w ) ∥ 2 , where the expectation is calculated with respect to the client index k.

Then, for a federated learning system where all N devices fully participate and the data are distributed independently and identically, the convergence rate of the FedDT algorithm is O ( 1/ NR ) , where R is the total number of iterations of stochastic gradient descent (SGD) performed by each client.

Proof of Proposition 3. Here, we provide a concise proof that primarily relies on the proof given by Qu et al. [49]. According to Equations (11) and (12), the loss function of the knowledge distillation module consists of three components:

For the task loss function, we adopt the cross-entropy loss, whose gradient is ∆Θ s F t . The distillation loss function is based on KL divergence, with the gradient ∆Θ s F d . The adaptive hidden loss function F (based on the mean squared error) has the gradient ∆Θ s F h .

In each iteration, the parameter update rule is

<!-- formula-not-decoded -->

where η is the learning rate.

According to convergence theorem of gradient descent, if the loss function F is convex and the learning rate η satisfies certain conditions (e.g., η ≤ 1 L , where L is the Lipschitz constant of F ), then lim r → + ∞ ∥ Θ r + 1 S -Θ ∗ S ∥ = 0, where Θ ∗ S is the optimal parameter.

For the loss function F in the knowledge distillation module, by analyzing its gradient, we obtain

<!-- formula-not-decoded -->

This indicates that the student model's parameters Θ S converge to the optimal solution Θ ∗ S as the number of iterations r increases. Let

<!-- formula-not-decoded -->

and the corresponding

<!-- formula-not-decoded -->

According to Proposition 2, we have

<!-- formula-not-decoded -->

So, based on Proposition 3, we let v = max k N ( | Dk | / ∑ N k = 1 | Dk | ) , κ = ( L / µ ) , γ = max { κ , E } , and we set the step size η r = ( 1/4 µ ( γ + r )) according to [49]. Then, based on Proposition 3, we have

<!-- formula-not-decoded -->

Then, according to Equations (54) and (55), we can obtain the convergence rate of FedDT in the following way:

$$
\begin{aligned}
\mathbb { E } ( J ( \hat { \Theta } _ { r } ^ { i } ) ) - J ^ { * } & \leq \frac { L } { 2 } \mathbb { E } \| \hat { \Theta } _ { r } - \Theta ^ { * } \| ^ { 2 } \\ & \leq 2 \hat { c } \| \Theta _ { 0 } - \Theta ^ { * } \| ^ { 2 } \left [ \frac { \kappa } { \mu } \frac { 1 } { R } \frac { 1 } { \gamma } \nu ^ { 2 } \delta ^ { 2 } + \frac { \kappa ^ { 2 } } { \mu } \frac { 1 } { ( R + \gamma ) ^ { 2 } } E ^ { 2 } G ^ { 2 } \right ] \\ & = O \left ( \frac { \kappa } { \mu } \frac { 1 } { R } \nu ^ { 2 } \delta ^ { 2 } + \frac { \kappa ^ { 2 } } { \mu } \frac { 1 } { R ^ { 2 } } E ^ { 2 } G ^ { 2 } \right )
\end{aligned}
$$

w h e r $ \hat { c

$$
$ is a sufficiently large constant for ineqaulity scaling $ \text { Therefore } if we set the local }
$$

where ˆ c is a sufficiently large constant for inequality scaling. Therefore, if we set the local error E = O ( ( R / N ) 1 2 ) , then O ( E 2 / R 2 ) = O ( 1/ NR ) , and the convergence rates of the federated averaging algorithm FedAvg and FedDT are both O ( 1/ NR ) .

## 6. Experiment

In this section, detail how we performed an empirical evaluation of the FedDT framework against leading FL algorithms, examining communication overhead, model performance, and convergence behavior. Moreover, ablation studies were carried out to quantitatively measure the individual contributions of each novel component. Finally, we systematically analyzed the underlying mechanisms driving the observed experimental outcomes.

## 6.1. Experimental Setup

## 6.1.1. Datasets

Two benchmark datasets, MNIST and Cifar10, which are widely used in classification tasks, were adopted as FL datasets, as shown in Table 2. These two datasets are widely used in the field of FL and provided us with rich and diverse data resources to train and test the performance of various FL algorithms.

(1) MNIST : The MNIST dataset consists of 10 categories (handwritten numbers 0 to 9); each image is a 28 × 28 pixel grayscale map. Because of its simple feature extraction mechanism, the MNIST dataset is widely adopted for training and validating small neural networks, making it a popular choice for educational purposes and algorithm benchmarking.

(2) Cifar10 : The Cifar10 dataset contains 10 distinct object categories represented as 32 × 32 pixel RGB images with three color channels. To address the challenges in feature extraction arising from the dataset's complexity, we applied data augmentation strategies such as random cropping and horizontal flipping to enrich training data diversity.

Table 2. Statistical information of common deep neural network compression and acceleration.

| Dataset   |   Class | Train    | Test     | Size    |
|-----------|---------|----------|----------|---------|
| MNIST     |      10 | 6 × 10 4 | 1 × 10 4 | 28 × 28 |
| Cifar10   |      10 | 5 × 10 4 | 1 × 10 4 | 32 × 32 |

## 6.1.2. Data Distribution

In the real world, data present a non-independent homogeneous distribution in different scenarios. The distribution characteristics, the diversity of the labels, the extent of the dataset critically affect the training performance and the model's applicability across diverse scenarios. Accordingly, the deliberate construction of data allocation schemes is fundamental for the comprehensive assessment of FL methodologies. To more realistically simulate the randomness of data and the non-independent homogeneous distribution of labels in the real world, this paper designed two data partitioning strategies, independent homogeneous distribution and non-independent homogeneous distribution, for the datasets.

(1) IID data : The entire dataset was first randomly disrupted, and then the same number of samples was uniformly sampled and distributed to 100 clients to maintain congruence between the local data distribution at each client and the global data distribution.

(2) Non-IID data : The aggregated dataset across all clients encompassed the complete dataset, yet the class distribution among clients remained uneven, with each client containing a subset of the total 10 classes. By employing labeling strategies, we could allocate Nc distinct classes to each client, where Nc represents the predetermined number of labeled classes per client.

## 6.1.3. Basic Configuration

Hardware Configuration: In evaluating the proposed FedDT, each client maintained communication solely with the server, prohibiting any form of information sharing among clients. Usually, 100 clients are used for simulation experiments. The training parameters are listed in Table 3 below.

Table 3. FL parameter settings.

| Parameter Symbol   | Description                             |   Setpoint |
|--------------------|-----------------------------------------|------------|
| K                  | Number of participating users per round |        100 |
| C                  | Percentage of participating users%      |         10 |
| E                  | Local iteration rounds                  |          5 |
| B                  | Local data batch size                   |         64 |
| lr                 | Learning rate                           |      0.001 |
| λ                  | Percentage of training samples%         |         80 |

## 6.1.4. Baselines

To verify the efficiency of the FedDT method's communication and the accuracy of model prediction, we chose three advanced FL baseline algorithms for comparison, including the classical FL algorithm FedAvg [7], and four efficient FL algorithms for data heterogeneity, Feddistll [42], FedKD [27], MOON [39], and FedProto [45]. The above three schemes are the best performing studies at present, so in this paper, we compared FedDT with the above three baseline methods.

(1) FedAvg is the first FL algorithm that systematically solves the distributed data training problem and has been applied in practice in production. The core idea is to achieve privacy protection through localized training, where the raw data remain stored locally on client devices, with only model parameter updates transmitted to the server.

(2) The Feddistill method combines two complementary federated augmentation (FAug) and federated distillation (FD) techniques, in which a GAN is used in federated augmentation to locally heterogeneous data augmented into independent and homogeneously distributed data, and federated distillation uploads logit vectors to reduce the communication overhead in FL caused by large-scale model parameters and frequent client-server interactions.

(3) The FedKD algorithm optimizes privacy leakage in the Feddistill method and further improves model performance and decreases communication overhead in heterogeneous FL via the combination of state-of-the-art FL integrating adaptive mutual knowledge distillation and dynamic gradient compression.

(4) MOON is a simple yet effective federated learning framework. Its core concept lies in addressing data heterogeneity by leveraging model-level contrastive learning, which corrects local training processes through the similarity between global model representations and local model representations, thereby mitigating the adverse effects of non-IID data distributions.

(5) FedProto replaces gradient communication with abstract class prototypes. Clients compute class prototypes based on their local data; the server aggregates these client prototypes to generate global prototypes. During training, clients align their local prototypes with global prototypes while minimizing classification loss.

## 6.2. Performance Analysis on the IID Dataset

## 6.2.1. Results on Centralized Machine Learning and FL

For a fair assessment of model performance, comparisons were made after 100 training iterations, ensuring that each model was trained on the same number of samples. In the case of MNIST, for example, there were 100 clients in the FL system, each with 600 samples, running 100 rounds of model training. Correspondingly, centralized model training used 60,000 samples to train 100 epochs. As the λ ratio remained bounded by 1, FL inherently employed a reduced training sample size compared to centralized learning frameworks.

Asummary of the accuracy levels achieved by the baseline models in the centralized machine learning and FL paradigms is provided in Table 4.

Table 4. Accuracy of different models on IID and non-IID Cifar10 datasets.

| Models                       | MLP   | CNN     |
|------------------------------|-------|---------|
| Dataset                      | MNIST | CIFAR10 |
| Optimizer                    | SGD   | Adam    |
| Basic accuracy (%)           | 97.13 | 84.73   |
| IID dataset accuracy (%)     | 96.20 | 83.77   |
| Non-IID dataset accuracy (%) | 47.94 | 48.05   |

## 6.2.2. Results on FedDT and Baseline Algorithms

In this section, a conventional IID FL scenario is analyzed, where all clients participated in training a common global model.

Figure 5 illustrates the convergence of test performance for six federated learning algorithms under IID data distributions over 100 communication rounds. The y-axis represents the model prediction accuracy, while the x-axis denotes communication rounds. As shown in Figure 5a, in the IID MNIST dataset, FedDT achieved an accuracy of 99.09%, comparable to that of FedKD. As training progressed, both FedDT and FedKD significantly outperformed the other four algorithms. In Figure 5b, it can be seen that in the IID Cifar10 dataset, FedDT achieved an accuracy of 87%, second only to MOON, and demonstrated clear superiority over the other four baseline methods.

(MNIST, Nc=10)

(Cifar10, Nc=10)

<!-- image -->

Accuracy

Figure 5. Convergence performance comparison of six algorithms across communication rounds under IID conditions. ( a ) Convergence performance comparison of six algorithms on the MNIST dataset under IID conditions. ( b ) Convergence performance of six algorithms on the Cifar10 dataset under IID conditions.

Table 5 summarizes the performance of various algorithms on MNIST/Cifar10 under standard federated learning settings. Figure 6 provides a visual comparison of the accuracy gaps between FedDT and other algorithms. FedDT achieved high accuracy levels of 99. 09% with MNIST and 87. 04% with Cifar10, underscoring its robustness and generalization capability in IID environments. These results validate that FedDT's multiround local update strategy enhances model performance through iterative local training.

Accuracy(%)

Figure 6. Comparison of accuracy of six algorithms on MNIST and Cifar10 datasets.

<!-- image -->

Table 5. Comparison of accuracy of different algorithms on various IID datasets in image classification tasks.

| Methods   |   MNIST (%) |   Cifar10 (%) |
|-----------|-------------|---------------|
| FedAvg    |       90.02 |         83.69 |
| Feddistll |       94.19 |         79.25 |
| FedKD     |       98.67 |         83.06 |
| MOON      |       95.67 |         88.23 |
| FedProto  |       92.37 |         82.56 |
| FedDT     |       99.09 |         87.04 |

## 6.2.3. Performance Analysis of Non-IID Data

For the non-independently identically distributed data scenario (non-IID), we proposed a category-controlled data partitioning strategy, as visualized in Figure 7. The vertical axis (y-axis) of the plot represents sample labels ranging from 0 to 9. Under the IID setting ( Nc = 10), each client was randomly allocated an equal share of 10 labeled categories via uniform sampling, maintaining the original IID distribution of both training and testing data. For non-IID scenarios, the data partitioning followed a deliberate stratification approach to generate class-imbalanced distributions across clients. For example, when Nc = 2, each client contained only two types of data labels, and there was no overlap of the data of clients. When Nc = 5, each client had five types of data labels, and at this time, there was some overlap in client data.

Figure 7. Data partitioning for clients when Nc = 10; Nc = 5; and Nc = 2. Bubble size indicates the amount of data. ( a ) Data partitioning scenario with Nc = 10, where client datasets were randomly sampled from the overall distribution. ( b ) Partitioning configuration with Nc = 5, allocating 5 class labels per client while permitting controlled data overlap. ( c ) Strict partitioning case with Nc = 2, assigning exactly 2 class labels per client with zero data overlap between clients.

<!-- image -->

Figure 8 depicts the convergence behavior of six federated learning algorithms on Cifar10 under non-IID settings ( Nc = 2 and Nc = 5) over 100 communication rounds. FedDT consistently outperformed the other five algorithms in both convergence speed and final accuracy. Table 6 details the performance of all algorithms on non-IID Cifar10 ( Nc = 2 and Nc = 5), while Figure 9 highlights the accuracy gap between FedDT and traditional methods. FedDT achieved state-of-the-art accuracy rates of 89.67%, 90.89%, 52.36%, and 74.41% on MNIST and Cifar10, surpassing all competitors.