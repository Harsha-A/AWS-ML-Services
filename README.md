# AWS-ML-Services
ML Services


# Comprehensive AWS Generative AI Exam Study Notes

This detailed guide covers all the AWS services tested on the generative AI certification exam, combining foundational concepts with exam-focused technical details.[1][2][3]

## Amazon Bedrock Core Services

### Amazon Bedrock
Fully managed service for building and scaling generative AI applications using foundation models (FMs) without infrastructure management.[3]

**Key Concepts:**
- Provides access to multiple pre-trained foundation models (GPT-based, BERT-based, Claude, Llama) for text generation, summarization, image creation, and code generation
- Supports fine-tuning of FMs with domain-specific data to adapt models for healthcare, finance, customer service without ML expertise
- Pay-as-you-go pricing based on token consumption; Provisioned Throughput option for guaranteed high-availability performance
- Integrates with S3, SageMaker, OpenSearch for end-to-end AI workflows and vector search capabilities
- Supports VPC integration for deploying models in isolated networks meeting compliance standards

**Security & Governance:**
- IAM integration for access control to models and data
- Encryption for data at rest and in transit
- CloudTrail and CloudWatch integration for auditing and monitoring model usage

### Amazon Bedrock AgentCore
Platform for building agentic AI systems that interact with external tools, maintain persistent memory, and execute secure actions.[4]

**Key Concepts:**
- Automates multi-step complex workflows by chaining tasks or models without manual intervention
- Manages data retrieval followed by generative output in customer service workflows
- Secure sandboxed code execution environment for agent actions
- Supports custom configurations for orchestrating tasks across different foundation models and external systems
- Error handling and monitoring capabilities for agent behavior tracking

### Amazon Bedrock Knowledge Bases
Enables retrieval-augmented generation (RAG) by connecting models with enterprise knowledge sources.[3]

**Key Concepts:**
- Supports document chunking strategies for optimal retrieval performance
- Uses Amazon Titan embeddings for vector representation of documents
- Integrates with vector databases (OpenSearch Serverless, Aurora PostgreSQL, Pinecone)
- Hybrid search combining keyword and semantic search capabilities
- Reduces hallucination by grounding model responses in real-world enterprise data

### Amazon Bedrock Prompt Management
Centralized service for designing, storing, versioning, and managing prompt templates.[3]

**Key Concepts:**
- Supports reusable prompt templates with variables for dynamic content
- Version control for prompt evolution and A/B testing
- Integration with Bedrock models for consistent prompt deployment
- Enables prompt optimization through iterative refinement and testing

### Amazon Bedrock Prompt Flows
Visual workflow orchestration for complex multi-step generative AI applications.

**Key Concepts:**
- Sequential prompt execution with conditional branching logic
- Chaining multiple model calls with intermediate processing steps
- Error handling and retry mechanisms for robust workflows
- Integration with Lambda functions for custom processing logic

### Prompt Engineering Techniques
Critical skill for maximizing model accuracy and relevance.[3]

**Techniques:**
- **Zero-shot**: No examples provided, model relies on pre-training
- **Few-shot**: Provide 2-5 examples to guide model behavior
- **Chain-of-thought**: Include reasoning steps in prompts for complex tasks
- Context optimization: Include relevant background information for domain-specific queries

## Amazon SageMaker Ecosystem

### Amazon SageMaker AI
Comprehensive, fully managed service for the entire ML lifecycle from data prep to model monitoring.[3]

**Core Capabilities:**
- Supports TensorFlow, PyTorch, MXNet, and other open-source frameworks
- Handles supervised, unsupervised, and reinforcement learning
- Provides managed infrastructure for training and deployment
- Integrates with EC2 spot instances for cost-optimized training

### SageMaker Unified Studio
Web-based IDE providing unified interface for all ML development tasks.[3]

**Features:**
- Single environment for data prep, training, debugging, and deployment
- Experiment tracking with artifact versioning and model comparison
- Collaboration tools for team-based ML development
- Notebook-based development with GPU/CPU instance selection

### SageMaker Data Wrangler
Simplifies data preparation with visual interface and 300+ built-in transformations.[3]

**Key Features:**
- Missing data handling (imputation, deletion strategies)
- Feature engineering (encoding, scaling, normalization)
- Data visualization for exploratory analysis
- Integration with S3, Redshift, RDS, Athena data sources
- Export transformed data to SageMaker Pipelines or Feature Store

### SageMaker Ground Truth
Creates high-quality training datasets through human and automatic labeling.[3]

**Capabilities:**
- Supports 2D/3D object detection, bounding boxes, semantic segmentation, text classification
- Active learning reduces labeling costs by auto-labeling simple tasks
- Integrated workforce options: Mechanical Turk, private workforce, vendor-managed teams
- Quality control through consensus-based validation and gold standard examples

### SageMaker Clarify
Detects bias in data/models and provides explainability for predictions.[3]

**Key Functions:**
- Pre-training bias detection across sensitive attributes (race, gender, age)
- Post-training bias metrics (disparate impact, demographic parity)
- SHAP (SHapley Additive exPlanations) for global and local model explainability
- Feature importance analysis for understanding model decisions
- Critical for regulatory compliance and responsible AI practices

### SageMaker Model Monitor
Automatically monitors deployed models for data drift and performance degradation.[3]

**Monitoring Capabilities:**
- Tracks accuracy, precision, recall, F1 score over time
- Detects input data distribution changes (data drift)
- Identifies prediction quality degradation
- Configurable CloudWatch alerts for anomaly detection
- Enables automated retraining triggers when thresholds are breached

### SageMaker JumpStart
Pre-built models and solutions for rapid ML project initiation.[3]

**Features:**
- Popular model architectures for NLP, computer vision, time series
- Pre-built solutions: fraud detection, demand forecasting, personalized recommendations, churn prediction
- One-click deployment of foundation models
- Fine-tuning capabilities for domain adaptation

### SageMaker Model Registry
Centralized repository for versioning, cataloging, and deploying ML models.[3]

**Capabilities:**
- Model versioning with metadata tracking (training data, parameters, metrics)
- Approval workflows for model promotion (dev → staging → production)
- Integration with CI/CD pipelines for automated deployment
- Model lineage tracking for audit and compliance

### SageMaker Neo
Optimizes ML models for deployment on edge devices and cloud instances.[3]

**Optimization Features:**
- Compiles models once, runs on multiple hardware platforms (ARM, Intel, NVIDIA)
- Reduces model size and improves inference performance (up to 2x speedup)
- Supports TensorFlow, PyTorch, MXNet, ONNX model formats
- Enables deployment on IoT devices, mobile, edge locations

### SageMaker Processing
Runs data processing and model evaluation workloads at scale.[3]

**Use Cases:**
- Feature engineering on large datasets
- Model evaluation with custom metrics
- Data validation before training
- Batch preprocessing for inference pipelines
- Supports scikit-learn, pandas, custom Docker containers

## Document AI and NLP Services

### Amazon Comprehend
Natural language processing service for text analysis.[3]

**Capabilities:**
- Entity extraction (people, places, organizations, dates)
- Sentiment analysis (positive, negative, neutral, mixed)
- Key phrase extraction for document summarization
- Language detection (100+ languages)
- Topic modeling for document categorization
- Custom entity recognition for domain-specific entities
- PII (Personally Identifiable Information) detection and redaction

### Amazon Textract
Automated document text and data extraction service.[3]

**Features:**
- OCR (Optical Character Recognition) for scanned documents
- Table extraction with cell-level accuracy
- Form parsing (key-value pair extraction)
- Signature detection in forms
- Query-based extraction for specific document sections
- Integration with A2I for human validation of low-confidence results

### Amazon Transcribe
Automatic speech recognition service converting audio to text.

**Key Features:**
- Real-time and batch transcription
- Custom vocabulary for domain-specific terminology
- Speaker identification (speaker diarization)
- Multi-language support with automatic language detection
- Profanity filtering and content redaction
- Medical transcription specialization for clinical documentation

## Computer Vision Services

### Amazon Rekognition
Image and video analysis using deep learning.[3]

**Capabilities:**
- Object and scene detection in images/videos
- Facial analysis (emotion, age range, gender)
- Face comparison and face search
- Celebrity recognition
- Text detection in images (OCR)
- Content moderation (unsafe content detection)
- Custom label training for domain-specific objects
- Integration with A2I for human review of uncertain predictions

## Enterprise Search and Knowledge Management

### Amazon Kendra
Intelligent enterprise search powered by ML for RAG applications.[2][5]

**Key Features:**
- Natural language query understanding with semantic search
- GenAI index optimized for retrieval augmented generation (RAG)
- Integrates with Amazon Q Business and Bedrock Knowledge Bases
- Connectors for 50+ data sources (SharePoint, S3, RDS, Salesforce, ServiceNow)
- Document ranking based on relevance and user permissions
- Faceted search with metadata filtering
- Learning from user interactions to improve results
- Access control list (ACL) support for secure document retrieval

## Conversational AI

### Amazon Lex
Low-code service for building conversational interfaces (chatbots, voice bots).[3]

**Features:**
- Natural language understanding (NLU) and automatic speech recognition (ASR)
- Multi-turn conversation management with context tracking
- Intent and slot recognition for extracting user requirements
- Integration with Lambda for business logic execution
- Multi-language support
- Sentiment analysis during conversations
- Voice and text channel support (phone, web, mobile, messaging platforms)

## Enterprise AI Assistants

### Amazon Q Business
AI-powered enterprise assistant for business workflows and knowledge management.

**Capabilities:**
- Natural language Q&A over enterprise data sources
- RAG-based responses grounded in company documents
- Integration with 40+ enterprise connectors
- User permission inheritance for secure information access
- Conversation history and context management
- Custom plugin development for business logic integration

### Amazon Q Business Apps
Low-code platform for building custom AI-powered business applications.[6]

**Features:**
- Drag-and-drop app builder without coding
- Generative AI capabilities for content creation
- Integration with Q Business for data access
- Custom workflow automation
- Shareable across organization with role-based access

### Amazon Q Developer
AI assistant for software development and AWS resource management.[7]

**Development Features:**
- Inline code completion and generation
- Security scanning for vulnerabilities
- Code explanation and documentation generation
- Debugging assistance and performance optimization
- Query AWS resources, architecture patterns, and documentation
- CLI integration for command generation

## Foundation Models

### Amazon Titan
Amazon's proprietary foundation models for text and embeddings.[3]

**Model Types:**
- **Titan Text**: Large language models for generation, summarization, Q&A
- **Titan Embeddings**: Convert text to vector embeddings for semantic search
- **Titan Image Generator**: Text-to-image generation and image editing
- Optimized for RAG workflows with high-quality embeddings
- Cost-effective compared to third-party models
- Built-in responsible AI guardrails

## Human-in-the-Loop AI

### Amazon Augmented AI (A2I)
Service for integrating human review workflows into ML predictions.[1]

**Key Concepts:**
- Human Loop workflow: Define conditions triggering human review (e.g., confidence < 80%)
- Direct integration with Textract, Rekognition, and SageMaker
- Workflow structure: HumanLoop → Review → Output Consolidation
- Private workforce or Mechanical Turk options

**Common Use Cases:**
- Content moderation requiring human judgment
- Low-confidence prediction validation
- PII redaction verification for compliance
- OCR output correction for medical/clinical documents
- Document classification review in large datasets
- Model drift monitoring through human feedback

## Exam Preparation Best Practices

**Focus Areas:**
- Master RAG architecture patterns with Bedrock Knowledge Bases and Kendra integration
- Understand prompt engineering techniques (zero-shot, few-shot, chain-of-thought)
- Learn SageMaker MLOps tools (Pipelines, Model Monitor, Clarify, Model Registry)
- Practice security configurations (VPC, IAM, encryption) for generative AI deployments
- Study cost optimization strategies (Provisioned Throughput vs on-demand, token management)
- Understand bias detection and model explainability with Clarify and A2I
- Know integration patterns between services (Bedrock + Kendra + Q Business)
- Review multi-modal model capabilities and use cases

**Evaluation Metrics:**
- ROUGE (text summarization quality)
- BLEU (translation accuracy)
- BERTScore (semantic similarity)
- Human evaluation criteria for generative outputs

This comprehensive guide covers the technical depth required for AWS generative AI certification success.[5][2][7][1][3]

[1](https://tutorialsdojo.com/amazon-augmented-ai-a2i/)
[2](https://docs.aws.amazon.com/kendra/latest/dg/what-is-kendra.html)
[3](https://aws.amazon.com/certification/certified-ai-practitioner/)
[4](https://aws.amazon.com/bedrock/agentcore/)
[5](https://www.xenonstack.com/blog/ai-agents-with-amazon-kendra)
[6](https://aws.amazon.com/blogs/training-and-certification/category/amazon-q/)
[7](https://www.pluralsight.com/paths/amazon-q-for-developer)
[8](https://docs.aws.amazon.com/augmented-ai/)
[9](https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-getting-started.html)
[10](https://notes.kodekloud.com/docs/AWS-Solutions-Architect-Associate-Certification/Services-Data-and-ML/Augmented-AI)
[11](https://dev.to/aws/have-you-heard-about-amazon-augmented-ai-434n)
[12](https://sagemaker-examples.readthedocs.io/en/latest/aws_marketplace/using_model_packages/amazon_augmented_ai_with_aws_marketplace_ml_models/amazon_augmented_ai_with_aws_marketplace_ml_models.html)



----------



# AWS Management and Governance Services for Generative AI Exam Study Notes

This guide provides detailed exam-focused coverage of AWS Management and Governance services critical for monitoring, optimizing, and managing generative AI workloads.[1][2][3][4]

## Monitoring and Observability

### Amazon CloudWatch
Comprehensive monitoring and observability service for AWS resources, applications, and ML models.[2][1]

**Core Capabilities:**
- Metrics repository collecting and storing performance data from AWS services and custom applications
- Default metrics for most AWS services (CPU, network, disk, status checks) without additional configuration
- Custom metrics for application-specific monitoring (model accuracy, token usage, inference latency)
- Metric resolution: Standard (5 minutes) or High-resolution (1 second)
- Metric retention: 15 months for standard resolution

**Key Features for AI/ML Workloads:**
- **SageMaker Integration**: Automatically collects ModelLatency, Invocations, InvocationsPerInstance, 4XXErrors, 5XXErrors metrics[5][2]
- **Bedrock Monitoring**: Tracks token consumption, API latency, throttling events, model invocation counts
- **Real-time Dashboards**: Visualize training metrics (loss, accuracy) in near real-time for ML experiments[5]
- **Percentile Statistics**: Track p50, p90, p99 latency for inference endpoints to understand tail performance
- **Anomaly Detection**: ML-powered anomaly detection for automatic baseline creation and alerting

**Alarms and Actions:**
- Create alarms on metric thresholds (e.g., ModelLatency > 500ms)
- Composite alarms combining multiple conditions (high latency AND high error rate)
- Alarm actions: SNS notifications, Auto Scaling policies, Lambda functions, Systems Manager automation

**CloudWatch for Cost Monitoring:**
- Track compute utilization (EC2, SageMaker instances) to identify underutilized resources
- Monitor Bedrock token consumption to optimize prompt engineering
- Set billing alarms for unexpected spending patterns

### Amazon CloudWatch Logs
Centralized log management for application, system, and service logs.[1][2]

**Key Concepts:**
- **Log Groups**: Container for log streams (e.g., `/aws/sagemaker/Endpoints/my-endpoint`)
- **Log Streams**: Sequence of log events from same source (e.g., individual container instances)
- **Log Events**: Individual log entries with timestamp and message
- Retention policies: 1 day to 10 years, or indefinite

**AI/ML Log Sources:**
- SageMaker training job logs (stdout/stderr from training containers)
- SageMaker endpoint invocation logs (request/response payloads)
- Bedrock API call logs (prompts, completions, metadata)
- Lambda function logs for custom processing logic
- Application logs from containerized AI services

**CloudWatch Logs Insights:**
- Interactive query language for analyzing log data
- Common queries: Error analysis, latency patterns, user behavior tracking
- Visualizations: Time series charts, bar graphs, tables
- Sample query for SageMaker errors: `fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc`

**Integration with AI Workflows:**
- Install CloudWatch Agent on EC2/ECS for custom application logs
- Enable SageMaker endpoint data capture for logging inference requests/responses
- Export logs to S3 for long-term archival and compliance
- Stream logs to Lambda for real-time processing and alerting

### Amazon CloudWatch Synthetics
Creates canaries that monitor endpoints and APIs with automated tests.[2]

**Key Features:**
- Scheduled synthetic transactions simulating user behavior
- HTTP/HTTPS endpoint monitoring with custom scripts (Node.js, Python)
- Visual monitoring: Screenshot capture for UI regression detection
- Canary types: Heartbeat (availability), API canary (response validation), Broken link checker, GUI workflow

**AI/ML Use Cases:**
- Monitor SageMaker real-time endpoint availability 24/7
- Validate Bedrock API response quality and latency
- Test end-to-end AI application workflows (user query → RAG → response)
- Alert on failures with CloudWatch Alarms integrated with SNS/Lambda

**Best Practices:**
- Create canaries calling inference endpoints with sample requests
- Set alarm thresholds for canary failure rate (e.g., >10% failures in 5 minutes)
- Use canary metrics to correlate availability issues with deployment changes

### AWS CloudTrail
Governance, compliance, and audit service tracking API activity across AWS account.[2]

**Core Functionality:**
- Records all API calls made to AWS services (console, CLI, SDK, automation tools)
- Captures: Identity (who), timestamp (when), source IP (where), action (what), resources (target)
- Event types: Management events (control plane), Data events (data plane like S3 object access)

**AI/ML Audit and Security:**
- Track who accessed or modified SageMaker models, Bedrock configurations, training data
- Monitor IAM role assumptions for ML workload security analysis
- Detect unauthorized API calls to sensitive AI services
- Compliance: Maintain immutable audit trail for regulatory requirements (HIPAA, GDPR, SOC 2)

**Integration:**
- Deliver logs to S3 for long-term storage and analysis
- Stream events to CloudWatch Logs for real-time monitoring
- Integrate with EventBridge for automated responses to specific API activities

**CloudTrail vs CloudWatch:**
- **CloudTrail**: Who did what, when? (API audit trail, governance)
- **CloudWatch**: How is the system performing? (metrics, logs, operational monitoring)

### Amazon Managed Grafana
Fully managed service for Grafana, providing data visualization and analytics dashboards.

**Key Features:**
- Pre-built dashboards for AWS services (CloudWatch, X-Ray, Prometheus)
- Multi-account and multi-region monitoring in unified interface
- Plugin ecosystem for extending visualization capabilities
- User authentication via AWS SSO, SAML, OAuth

**AI/ML Monitoring:**
- Visualize SageMaker training metrics across multiple experiments
- Compare model performance over time with custom queries
- Monitor distributed training cluster resource utilization
- Create operational dashboards combining CloudWatch metrics, logs, and traces

## Auto Scaling and Automation

### AWS Auto Scaling
Unified scaling for multiple AWS services to optimize performance and cost.

**Supported Services for AI/ML:**
- SageMaker endpoint instances (inference auto-scaling)
- EC2 instances for distributed training clusters
- ECS/Fargate tasks for containerized AI applications
- DynamoDB tables for vector store scaling

**SageMaker Endpoint Auto Scaling:**
- Target tracking: Scale based on InvocationsPerInstance metric
- Configure min/max instance counts and target metric value
- Scale-out cooldown: Wait time before adding more instances (default 300s)
- Scale-in cooldown: Wait time before removing instances (default 300s)
- Protects against rapid scaling fluctuations and cost spikes

**Best Practices:**
- Set conservative scale-in cooldown for ML endpoints (5-10 minutes) to avoid cold starts
- Monitor scaling activities in CloudWatch to optimize thresholds
- Use scheduled scaling for predictable traffic patterns (e.g., business hours)
- Combine with Provisioned Throughput for Bedrock to guarantee capacity during high demand

### AWS Systems Manager
Unified interface for operational data and automation across AWS resources.

**Key Components:**
- **Parameter Store**: Secure, hierarchical storage for configuration data and secrets (API keys, model URIs)
- **Session Manager**: Secure shell access to EC2/on-premises instances without SSH keys
- **Patch Manager**: Automated OS patching for EC2 fleets
- **Run Command**: Execute commands on multiple instances simultaneously
- **State Manager**: Maintain consistent instance configurations

**AI/ML Use Cases:**
- Store and version ML model hyperparameters in Parameter Store
- Automate SageMaker training job submission via Run Command
- Maintain consistent software dependencies on training instance fleets
- Secure access to notebook instances without exposing SSH ports

## Cost Management

### AWS Cost Explorer
Visualize, understand, and manage AWS costs and usage over time.[3]

**Core Features:**
- Interactive charts showing cost trends by service, region, usage type
- Filtering and grouping by multiple dimensions (service, tag, instance type)
- Forecasting: Predict future costs based on historical patterns
- Cost allocation tags for tracking AI project spending
- Rightsizing recommendations for underutilized resources

**AI/ML Cost Analysis:**
- Identify expensive services (SageMaker training, Bedrock token usage, S3 storage)
- Compare training costs across different instance types (ml.p4d vs ml.p3)
- Track Bedrock API costs by model (Claude vs Titan) to optimize model selection
- Analyze cost trends after implementing optimization strategies

**Reservations and Savings Plans:**
- SageMaker Savings Plans: Up to 64% discount for committed usage
- EC2 Savings Plans: Cover training/inference instance costs
- Reserved Capacity: Guarantee SageMaker notebook/endpoint instance availability

### AWS Cost Anomaly Detection
ML-powered service identifying unusual spending patterns and root causes.[6][3]

**How It Works:**
- Uses machine learning to establish spending baselines across services
- Continuously monitors actual spend against predicted patterns
- Detects anomalies using rolling 24-hour windows for faster identification[3]
- Sends alerts via email, SNS, or Slack when anomalies detected

**Enhanced Detection Algorithm (Nov 2025):**
- Compares current costs against equivalent 24-hour periods from previous days
- Removes delay from incomplete calendar-day comparisons
- Accounts for workloads with different morning/evening usage patterns[3]
- Reduces false positives by contextual time-of-day analysis

**Configuration:**
- Define cost monitors: Entire account, specific services, or cost allocation tags
- Set alert threshold: Dollar amount ($100) or percentage (20% increase)
- Choose notification channels: Email, SNS topics for Slack/PagerDuty integration
- Segment by: Service, linked account, cost category, tag

**AI/ML Cost Anomaly Scenarios:**
- Unexpected SageMaker training job costs from instance type misconfiguration
- Bedrock token usage spikes from inefficient prompts or RAG loops
- S3 storage growth from unmanaged training data or model artifacts
- Inference endpoint running 24/7 instead of on-demand schedule

**Limitations:**
- Requires 10-14 days of usage data to establish baseline
- Manual configuration of monitors and segments
- No unit cost analysis (per-customer, per-project granularity)[6]
- Best for gross anomalies, not fine-grained cost optimization

## Communication and Collaboration

### AWS Chatbot
Interactive agent enabling ChatOps for AWS services via Slack, Microsoft Teams, Amazon Chime.

**Key Features:**
- Receive CloudWatch alarms, AWS Health notifications, Security Hub findings in chat
- Execute AWS CLI commands directly from chat (read-only or admin actions)
- Configure notification routing by severity, service, or tag
- IAM role-based permissions control chat command execution

**AI/ML ChatOps:**
- Receive alerts when SageMaker training jobs fail or complete
- Get notified of Bedrock API throttling or quota limits
- Query CloudWatch metrics from chat: `@aws cloudwatch get-metric-statistics`
- Acknowledge and resolve incidents collaboratively in team channels

**Security Considerations:**
- Use least-privilege IAM roles for Chatbot
- Enable audit logging of commands executed via chat
- Restrict admin actions to specific channels or users

## Service Management

### AWS Service Catalog
Centralized governance for IT services, enabling standardized provisioning of approved resources.

**Key Concepts:**
- **Products**: CloudFormation templates for AWS resources (e.g., SageMaker notebook with approved configuration)
- **Portfolios**: Collections of products with access controls
- **Constraints**: Rules limiting product configuration (instance types, regions)
- **Provisioned Products**: Launched instances of catalog products

**AI/ML Governance:**
- Create approved SageMaker notebook configurations with pre-installed libraries
- Standardize training job templates with security controls (VPC, encryption)
- Enforce cost guardrails by limiting instance types (ml.m5 family only)
- Provide self-service access to ML infrastructure without granting direct IAM permissions

**Benefits:**
- Consistent infrastructure deployment across teams
- Centralized version control for ML environment templates
- Compliance enforcement through constraints and launch rules
- Audit trail of provisioned resources for governance

## AWS Well-Architected Tool

### Core Framework
Provides architectural best practices across six pillars for evaluating workloads.[7][4]

**Six Pillars:**
1. **Operational Excellence**: Monitor, operate, and continuously improve processes
2. **Security**: Protect information, systems, and assets
3. **Reliability**: Recover from failures, scale to meet demand
4. **Performance Efficiency**: Use resources efficiently to meet requirements
5. **Cost Optimization**: Achieve business outcomes at lowest price point
6. **Sustainability**: Minimize environmental impact of cloud workloads

### Generative AI Lens
Specialized lens extending Well-Architected Framework for generative AI applications.[4]

**Operational Excellence for GenAI:**
- Achieve consistent model output quality through evaluation frameworks (ROUGE, BLEU, human feedback)
- Monitor operational health: Token usage, latency, error rates, model drift
- Maintain traceability: Log prompts, responses, and model versions for debugging
- Automate lifecycle management: CI/CD pipelines for model deployment and updates
- Determine when to execute model customization: Fine-tuning vs prompt engineering decisions

**Security for GenAI:**
- Protect endpoints: VPC isolation, encryption in transit/at rest, IAM least privilege
- Mitigate harmful outputs: Content filtering, guardrails, human review workflows (A2I)
- Monitor and audit events: CloudTrail API logging, CloudWatch metrics for anomalous behavior
- Secure prompts: Prevent prompt injection attacks, validate user inputs
- Remediate model poisoning risks: Data validation, provenance tracking, regular retraining

**Reliability for GenAI:**
- Handle throughput requirements: Auto-scaling, Provisioned Throughput for Bedrock
- Maintain reliable component communication: Retry logic, circuit breakers, graceful degradation
- Implement observability: Distributed tracing with X-Ray, structured logging
- Handle failures gracefully: Fallback models, cached responses, error messaging
- Version artifacts: Model registry, prompt versioning, dataset lineage

**Performance Efficiency for GenAI:**
- Capture and improve model performance: A/B testing, continuous evaluation
- Maintain acceptable performance: Latency budgets, caching strategies, batch processing
- Optimize computation resources: Instance type selection, model quantization, SageMaker Neo
- Improve data retrieval: Vector store optimization, hybrid search, chunking strategies for RAG

**Cost Optimization for GenAI:**
- Select cost-optimized models: Compare cost per token across Bedrock models
- Balance cost and performance of inference: Provisioned vs on-demand, batch vs real-time
- Engineer prompts for cost: Minimize token count, use caching, avoid redundant context
- Optimize vector stores: Right-size databases, implement TTL policies for embeddings
- Optimize agent workflows: Reduce tool invocations, implement result caching

**Sustainability for GenAI:**
- Minimize computational resources for training: Transfer learning, efficient architectures
- Optimize customization: Use parameter-efficient fine-tuning (PEFT) instead of full fine-tuning
- Reduce hosting footprint: Model quantization, instance rightsizing, auto-scaling policies
- Efficient data processing: Incremental updates, deduplication, compression
- Leverage serverless: Lambda for intermittent workloads, Bedrock for managed inference

### Using the Well-Architected Tool

**Review Process:**
1. Define workload in AWS WA Tool (name, description, environment)
2. Apply relevant lenses (AWS Well-Architected Framework + Generative AI Lens)
3. Answer questions across all pillars with team collaboration
4. Identify high/medium risk issues (HRIs/MRIs) based on best practice gaps
5. Generate improvement plan with prioritized remediation actions
6. Track progress over time with milestone comparisons

**Generative AI Lens Availability:**
- Download from AWS Well-Architected custom lens GitHub repository
- Import as custom lens into AWS WA Tool
- Available for all AWS accounts at no additional charge

## Exam Preparation Focus Areas

**Monitoring Best Practices:**
- Know which CloudWatch metrics are critical for ML endpoint monitoring (ModelLatency, 5XX errors)[2]
- Understand when to use CloudWatch Logs vs CloudTrail vs X-Ray
- Configure CloudWatch Alarms with appropriate thresholds and actions
- Implement CloudWatch Synthetics canaries for endpoint availability testing

**Cost Optimization Strategies:**
- Use Cost Anomaly Detection for proactive spending alerts[3]
- Leverage Cost Explorer for historical analysis and forecasting
- Apply Savings Plans and Reserved Capacity for predictable workloads
- Tag resources consistently for cost allocation and chargeback

**Automation and Governance:**
- Implement Service Catalog for standardized ML environment provisioning
- Use Systems Manager Parameter Store for configuration management
- Configure Auto Scaling for SageMaker endpoints with appropriate cooldown periods
- Apply Well-Architected Tool reviews regularly for continuous improvement[4]

**Security and Compliance:**
- Enable CloudTrail for comprehensive API audit logging
- Implement least-privilege IAM policies for ML workloads
- Use VPC endpoints for private connectivity to AWS services
- Apply encryption at rest and in transit for all AI/ML data

This comprehensive guide covers the management and governance services essential for operating production generative AI workloads on AWS according to best practices.[1][4][2][3]

[1](https://tutorialsdojo.com/amazon-cloudwatch/)
[2](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/AWS-Machine-Learning-Associate-Practice-Exams)
[3](https://aws.amazon.com/about-aws/whats-new/2025/11/aws-cost-anomaly-detection-accelerates-anomaly/)
[4](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/generative-ai-lens.html)
[5](https://aws.amazon.com/blogs/machine-learning/use-amazon-cloudwatch-custom-metrics-for-real-time-monitoring-of-amazon-sagemaker-model-performance/)
[6](https://spot.io/resources/aws-cost-optimization/aws-cost-anomaly-detection-pros-cons-and-how-to-get-started/)
[7](https://docs.aws.amazon.com/wellarchitected/latest/userguide/lenses.html)
[8](https://docs.aws.amazon.com/machine-learning/latest/dg/cw-doc.html)
[9](https://aws.amazon.com/blogs/training-and-certification/category/management-tools/amazon-cloudwatch/)
[10](https://www.geeksforgeeks.org/cloud-computing/introduction-to-amazon-cloudwatch/)
