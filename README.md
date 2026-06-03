Abstract

	Knowledge Tracing (KT) aims to model the dynamic evolution of learners' knowledge states
	and predict their future performance. Existing models have improved either temporal 
	dependency modeling or knowledge association modeling, yet they still lack a unified 
	framework that can simultaneously emphasize critical local learning events and integrate 
	information across multiple temporal scales. To address this issue, this paper proposes 
	a Dynamic Multi-scale Knowledge Tracing model (DMKT). The model is cognitively inspired 
	by two complementary ideas in learning: selective focus on critical events and multi-scale 
	integration of historical information. Specifically, we design a dynamic local sampling 
	(DePatch) module that employs a lightweight offset prediction network to dynamically 
	locate and extract key local segments during the learning process. The offset predictor 
	first partitions the input embedding sequence into overlapping patches through an unfold 
	operation, then independently processes each channel of every patch via depthwise separable 
	convolutions to predict center and scale offsets. In addition, we employ convolution kernels 
	of different scales to extract multi-scale information in parallel within the sequence, and 
	leverage learnable decay rates and causal time-distance matrices to compute decay weights. 
	This mechanism captures short-, medium-, and long-range temporal patterns while modeling the 
	gradual attenuation of historical information. The experimental results on four benchmark datasets 
	demonstrate that DMKT outperforms existing strong baselines. This validates that its unified 
	framework, which simulates selective focus and multi-scale memory integration, effectively 
	advances knowledge tracing towards a more cognitively aligned paradigm.
	
