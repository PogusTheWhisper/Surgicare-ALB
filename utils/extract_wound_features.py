import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import warnings
from transformers.utils import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

class CLIPWoundFeatureExtractor:
    """Feature extractor using BiomedVLP-BioViL-T for wound descriptions."""

    MODEL_NAME = "microsoft/BiomedVLP-BioViL-T"

    CLASS_FEATURES_EN = {
        'Abrasions': [
            "Superficial wound limited to the epidermis.",
            "Redness and raw appearance typical of abrasions.",
            "Minimal exudate with intact surrounding skin.",
            "Scab formation is present as part of healing.",
            "The wound surface appears scraped with irregular borders.",
            "There is mild serous fluid visible on the wound.",
            "No signs of necrosis or infection are observed.",
            "Periwound skin shows mild inflammation and dryness.",
            "Wound area is shallow and does not extend into dermis.",
            "Granulation tissue is forming on the wound bed."
        ],
        'Bruises': [
            "Bluish-purple discoloration of the skin.",
            "No open wound or tissue loss is observed.",
            "Tenderness and swelling without active bleeding.",
            "Discoloration fades from purple to yellow over time.",
            "Skin remains intact with localized subdermal bleeding.",
            "The area is soft to the touch but non-blanching.",
            "Swelling is present without any exudate or odor.",
            "No signs of epidermal disruption or trauma.",
            "Periwound skin is normal in texture but discolored.",
            "Bruised area is tender and warm without signs of infection."
        ],
        'Burns': [
            "Blister formation consistent with second-degree burns.",
            "Charred or leathery skin appearance indicating full-thickness burn.",
            "Redness and swelling in the burn area.",
            "Burn site has dry, cracked skin with sloughing.",
            "Peeling skin with visible dermal exposure.",
            "The wound emits a faint odor typical of burn injuries.",
            "No significant bleeding, but eschar is forming.",
            "Surrounding skin is erythematous and painful.",
            "Burn depth extends into subcutaneous layers.",
            "Burn margins are irregular with variable tissue response."
        ],
        'Cut': [
            "Well-defined linear wound with clean edges.",
            "Bleeding is present, consistent with a fresh cut.",
            "Exposed tissue at the wound site.",
            "The wound edges are approximated with signs of healing.",
            "There is minor clotting along the wound bed.",
            "Cut extends partially through the dermis.",
            "Moderate bleeding occurred but is now controlled.",
            "No signs of infection or necrosis around the laceration.",
            "Wound bed is moist with early granulation tissue.",
            "Cut was likely caused by a sharp object."
        ],
        'Normal': [
            "Skin appears intact with no visible injury.",
            "There is no redness, swelling, or open wound.",
            "Normal skin tone and texture with no exudate.",
            "The area is dry and free of abnormalities.",
            "No signs of trauma, bruising, or inflammation.",
            "Skin is smooth with no lesions or discoloration.",
            "Temperature and texture are consistent with healthy tissue.",
            "No maceration or irritation present.",
            "Healthy skin integrity is maintained.",
            "No pain, odor, or drainage reported in the area."
        ]
    }
    
    CLASS_FEATURES_TH = {
        'Abrasions': [
            "แผลถลอกอยู่เพียงชั้นหนังกำพร้าเท่านั้น",
            "มีรอยแดงและลักษณะเหมือนถลอกที่พบได้ทั่วไป",
            "มีของเหลวเล็กน้อยและผิวหนังรอบแผลยังคงสมบูรณ์",
            "พบสะเก็ดแผลซึ่งเป็นส่วนหนึ่งของกระบวนการหาย",
            "พื้นผิวแผลดูเหมือนถูกขูดขีด ขอบแผลไม่เรียบ",
            "มีของเหลวใสเล็กน้อยปรากฏบนแผล",
            "ไม่พบสัญญาณของเนื้อตายหรือการติดเชื้อ",
            "ผิวหนังรอบแผลมีการอักเสบเล็กน้อยและแห้ง",
            "แผลมีความตื้นและไม่ลึกถึงชั้นหนังแท้",
            "พบเนื้อเยื่อแกรนูลกำลังก่อตัวในบริเวณแผล"
        ],
        'Bruises': [
            "มีรอยช้ำสีน้ำเงินอมม่วงบนผิวหนัง",
            "ไม่พบแผลเปิดหรือการสูญเสียเนื้อเยื่อ",
            "บริเวณที่ช้ำมีอาการบวมและเจ็บแต่ไม่มีเลือดออก",
            "สีของรอยช้ำเปลี่ยนจากม่วงเป็นเหลืองตามเวลา",
            "ผิวหนังยังคงสมบูรณ์แต่มีเลือดออกใต้ผิวหนังเฉพาะจุด",
            "บริเวณนี้นิ่มเมื่อสัมผัสแต่ไม่ซีดลงเมื่อกด",
            "มีอาการบวมแต่ไม่มีของเหลวหรือกลิ่นผิดปกติ",
            "ไม่พบร่องรอยของการฉีกขาดหรือบาดเจ็บภายนอก",
            "ผิวหนังรอบแผลมีสีผิดปกติแต่ลักษณะพื้นผิวปกติ",
            "บริเวณที่ช้ำรู้สึกเจ็บและอุ่นโดยไม่มีสัญญาณของการติดเชื้อ"
        ],
        'Burns': [
            "พบบวมพองซึ่งสอดคล้องกับแผลไฟไหม้ระดับที่สอง",
            "ผิวหนังไหม้ดำหรือแข็งคล้ายหนัง บ่งชี้แผลไหม้ลึกถึงชั้นผิวหนังทั้งหมด",
            "มีรอยแดงและบวมในบริเวณที่ไหม้",
            "บริเวณแผลไหม้มีผิวแห้ง แตก และลอกออก",
            "ผิวหนังลอกออกและเห็นชั้นผิวหนังด้านใน",
            "แผลมีกลิ่นจางๆ ซึ่งพบได้ในแผลไฟไหม้",
            "ไม่มีเลือดออกมาก แต่มีสะเก็ดแผลเริ่มก่อตัว",
            "ผิวหนังรอบแผลมีรอยแดงและเจ็บปวด",
            "ความลึกของแผลไหม้ลามถึงชั้นใต้ผิวหนัง",
            "ขอบแผลไหม้ไม่เรียบและมีปฏิกิริยาของเนื้อเยื่อแตกต่างกัน"
        ],
        'Cut': [
            "แผลมีลักษณะเป็นเส้นตรง ขอบแผลชัดเจน",
            "มีเลือดออก สอดคล้องกับแผลที่เพิ่งเกิดใหม่",
            "เห็นเนื้อเยื่อภายในบริเวณแผล",
            "ขอบแผลเริ่มติดกัน บ่งบอกกระบวนการสมานแผล",
            "มีการแข็งตัวของเลือดเล็กน้อยในแผล",
            "แผลลึกลงถึงชั้นหนังแท้บางส่วน",
            "เคยมีเลือดออกปานกลาง แต่ขณะนี้ควบคุมได้แล้ว",
            "ไม่พบการติดเชื้อหรือเนื้อตายรอบแผล",
            "พื้นแผลมีความชื้น และเริ่มมีเนื้อเยื่อแกรนูล",
            "แผลดูเหมือนเกิดจากของมีคม"
        ],
        'Normal': [
            "ผิวหนังดูปกติและไม่มีบาดแผลให้เห็น",
            "ไม่พบรอยแดง บวม หรือแผลเปิด",
            "สีผิวและลักษณะพื้นผิวปกติ ไม่มีของเหลวผิดปกติ",
            "บริเวณนี้แห้งและไม่พบความผิดปกติใดๆ",
            "ไม่พบร่องรอยของการบาดเจ็บ รอยช้ำ หรือการอักเสบ",
            "ผิวเรียบ ไม่มีแผลหรือจุดผิดปกติ",
            "อุณหภูมิและลักษณะผิวสอดคล้องกับผิวหนังปกติ",
            "ไม่พบการชื้นแฉะหรือการระคายเคืองบริเวณนี้",
            "โครงสร้างของผิวหนังยังคงแข็งแรงและสมบูรณ์",
            "ไม่พบอาการปวด กลิ่น หรือของเหลวใดๆ จากบริเวณนี้"
        ]
    }



    SIMILARITY_THRESHOLD = 0.3  # Ignore weak matches below this

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model.eval().to(self.device)

    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Tokenize and encode text prompts."""
        with torch.no_grad():
            tokens = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=prompts,
                add_special_tokens=True,
                padding='longest',
                return_tensors='pt'
            ).to(self.device)

            text_embeds = self.model.get_projected_text_embeddings(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask
            )
            return text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Get image embeddings from the model."""
        raise NotImplementedError("This model does not currently support image feature projection in this script.")

    def extract_features(
        self,
        image_path: str,
        wound_class: str,
        top_k: int = 10,
        lang: str = 'en'
    ) -> List[Tuple[str, float]]:
        """
        Extract features using English prompts for accuracy, return in selected language.
        """
        en_prompts = self.CLASS_FEATURES_EN.get(wound_class, [])
        th_prompts = self.CLASS_FEATURES_TH.get(wound_class, [])

        if not en_prompts:
            print(f"No English prompts available for class '{wound_class}'")
            return []

        try:
            # Encode English prompts for similarity
            text_embeds = self._encode_text(en_prompts)
            similarities = torch.mm(text_embeds, text_embeds.t())
            ref_sim = similarities[0]  # Use first as reference (placeholder)

            results = [(i, en_prompts[i], ref_sim[i].item()) for i in range(len(en_prompts))]
            filtered = [item for item in results if item[2] >= self.SIMILARITY_THRESHOLD]
            sorted_features = sorted(filtered, key=lambda x: x[2], reverse=True)[:top_k]

            # Replace with Thai if lang == 'th' and Thai prompts exist
            if lang == 'th' and th_prompts:
                return [(th_prompts[i], score) for i, _, score in sorted_features]
            else:
                return [(text, score) for _, text, score in sorted_features]

        except Exception as e:
            print(f"Error during feature extraction: {e}")
            return []


    def get_all_features(self) -> Dict[str, List[str]]:
        """Return all feature prompts grouped by class."""
        return self.CLASS_FEATURES
