Here is a breakdown of **models used in each agent**, specifying whether they need training or can use pre-trained weights, and how to train or fine-tune them if needed:

---

## 🧠 Final Model List per Agent

### 1. **Detector Agent (DA)**

**Task:** Detect tumor ROIs with confidence.

| Component      | Model                       | Pretrained?     | Action Needed                                                       |
| -------------- | --------------------------- | --------------- | ------------------------------------------------------------------- |
| Backbone       | YOLOv8-nano                 | ✅ Yes (on COCO) | ⚠️ Fine-tune on annotated brain MRI dataset (e.g., BraTS or custom) |
| Detection Head | Custom tumor detection head | ❌ No            | 🔧 Train from scratch using bounding box labels on tumor locations  |

**Training:**

* Format MRI slices as 2D grayscale images or 3-channel pseudo-color.
* Annotate bounding boxes for tumors.
* Use Ultralytics YOLOv8 fine-tuning pipeline with domain-specific configs.

---

### 2. **Refiner Agent (RA)**

**Task:** Pixel-level segmentation and boundary refinement.

| Component | Model     | Pretrained? | Action Needed                                        |
| --------- | --------- | ----------- | ---------------------------------------------------- |
| Encoder   | MobileViT | ✅ Yes       | 🔧 Fine-tune on BraTS or custom segmentation dataset |
| Decoder   | U-Net     | ❌ No        | 🔧 Train together with encoder on segmentation maps  |

**Training:**

* Use 2D slice-based or 3D patch-based training from BraTS.
* Use Dice + BCE loss for segmentation.
* Train end-to-end or encoder-frozen depending on data size.

---

### 3. **Validator Agent (VA)**

**Task:** Ensemble-based uncertainty estimation and validation.

| Component  | Model                           | Pretrained?             | Action Needed                                    |
| ---------- | ------------------------------- | ----------------------- | ------------------------------------------------ |
| Ensemble 1 | SegFormer (small)               | ✅ Yes                   | ⚠️ Fine-tune on segmentation masks               |
| Ensemble 2 | DeepLabV3+ (MobileNet backbone) | ✅ Yes                   | ⚠️ Fine-tune or lightweight retraining           |
| Ensemble 3 | nnU-Net                         | ✅ Yes (config template) | ⚠️ Train on your dataset using nnU-Net framework |

**Training:**

* Use 3–5 fold cross-validation to generate disagreement maps.
* Compute epistemic uncertainty using ensemble variance.

---

### 4. **Explainer Agent (EA)**

**Task:** Generate visual and textual explanations.

| Component      | Model                    | Pretrained? | Action Needed                                 |
| -------------- | ------------------------ | ----------- | --------------------------------------------- |
| Vision Encoder | CLIP (ViT-B/32)          | ✅ Yes       | Use as-is to embed MRI + masks                |
| LLM            | Clinical LLaMA-7B + LoRA | ✅ Partially | 🔧 LoRA fine-tuning on radiology reports      |
| Adapter        | Visual → Text bridge     | ❌ No        | 🔧 Train lightweight adapter for CLIP → LLaMA |

**Training:**

* Create \<image, explanation> pairs from BraTS reports or mimic clinical data.
* Fine-tune using PEFT (LoRA) on GPTQ or INT8 quantized LLaMA for efficiency.
* Use contrastive loss + supervised explanation generation.

---

### 5. **Coordinator Agent (CA) – ReAct Controller (Gemini 1.5 Pro)**

**Task:** Orchestrates entire pipeline using ReAct framework.

| Component  | Model                 | Pretrained?      | Action Needed                                        |
| ---------- | --------------------- | ---------------- | ---------------------------------------------------- |
| LLM        | Gemini 1.5 Pro        | ✅ Yes            | Use via API – no training                            |
| Scratchpad | Internal ReAct memory | ✅ Yes (internal) | Maintain structured thoughts/actions/logs in prompts |

**Training:**

* ❌ Not required.
* ✅ Prompt engineering only.

---

## 🔄 Summary Table

| Agent  | Model(s)                                  | Needs Training?   | Usage                               |
| ------ | ----------------------------------------- | ----------------- | ----------------------------------- |
| **DA** | YOLOv8-nano + custom head                 | ✅ Yes             | Fine-tune on bounding boxes         |
| **RA** | MobileViT + U-Net                         | ✅ Yes             | Fine-tune on masks                  |
| **VA** | Ensemble (SegFormer, DeepLabV3+, nnU-Net) | ✅ Yes             | Train ensemble for uncertainty      |
| **EA** | CLIP + Clinical LLaMA + LoRA              | ✅ Yes (LoRA only) | Use CLIP as-is, LoRA-finetune LLaMA |
| **CA** | Gemini 1.5 Pro                            | ❌ No              | Use via API with ReAct prompts      |

---

