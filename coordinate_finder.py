# coordinate_finder.py (OCR強化版)

import os
import base64
import json
from typing import Optional, Dict, List, Tuple

from PIL import Image
import pyautogui
import pytesseract
import easyocr
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import cv2 # <-- OpenCVをインポート

# --- 設定 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("EasyOCRリーダーを初期化しています...")
easyocr_reader = easyocr.Reader(['ja', 'en'])
print("EasyOCRリーダーの初期化が完了しました。")


# --- Helper Functions (変更なし) ---
def take_screenshot(filename: str = "screenshot.png") -> str:
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"DEBUG: スクリーンショットを {filename} として保存しました。")
    return filename

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_combined_bbox(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    if not bboxes:
        return (0, 0, 0, 0)
    min_x = min(b[0] for b in bboxes)
    min_y = min(b[1] for b in bboxes)
    max_r = max(b[0] + b[2] for b in bboxes)
    max_b = max(b[1] + b[3] for b in bboxes)
    return (min_x, min_y, max_r - min_x, max_b - min_y)


# --- Core Logic Functions (Phase 1, 3は変更なし, Phase 2を刷新) ---

def phase1_get_region_with_llm(image_path: str, target_text: str, context_hint: str) -> Optional[Dict[str, int]]:
    # (この関数は変更なし)
    print("\n--- Phase 1: LLMによる領域予測 ---")
    base64_image = encode_image_to_base64(image_path)
    prompt = f"""
    あなたは優秀なUIアナリストです。あなたの仕事は、スクリーンショット画像から特定のUI要素が存在する「領域」を特定することです。
    **指示:** 添付された画像の中から、「{target_text}」というテキストを含むUI要素を探してください。
    **コンテキスト:** {context_hint}
    **重要なルール:**
    1. あなたの目的は、テキストそのものの正確な座標を見つけることではありません。後続のOCRがスキャンするための「探索エリア」を定義することです。
    2. したがって、返すバウンディングボックスは、テキスト単体ではなく、それが含まれるUIコンポーネント全体（例：メニューバー全体、ボタン全体、ツールバー全体）を囲むような**広めの領域**にしてください。
    3. たとえ本文中に同じ単語があっても、コンテキストに合致するUI要素の方を優先してください。
    **出力形式:** {{ "x": <int>, "y": <int>, "width": <int>, "height": <int> }} このJSON形式のみで、他の説明は一切不要です。
    """
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"},},],}], max_tokens=100, response_format={"type": "json_object"})
        content = response.choices[0].message.content
        print(f"DEBUG: LLMからの応答: {content}")
        region = json.loads(content)
        if all(k in region for k in ["x", "y", "width", "height"]):
            print(f"INFO: 予測された領域: {region}")
            return region
        else:
            print("ERROR: LLMからのJSONに必要なキーが含まれていません。")
            return None
    except Exception as e:
        print(f"ERROR: Phase 1でエラーが発生しました: {e}")
        return None

# ▼▼▼▼▼ ここからが刷新されたOCR処理 ▼▼▼▼▼
def advanced_ocr_scan(image: np.ndarray, ocr_engine: str, scale_factor: int = 2) -> List[Dict]:
    """
    指定された画像に対して、拡大・二値化・白黒反転などの前処理を施し、高精度なOCRを実行する。
    """
    # 1. 画像の前処理（グレースケール化と拡大）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    resized_gray = cv2.resize(gray, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # 2. 通常画像と白黒反転画像の二値化バージョンを作成
    _, binary_normal = cv2.threshold(resized_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_inverted = cv2.bitwise_not(binary_normal)
    
    image_versions = {
        "normal": binary_normal,
        "inverted": binary_inverted
    }

    all_results = []
    for version_name, img_to_scan in image_versions.items():
        print(f"DEBUG: OCRスキャン中 ({version_name} version)...")
        
        # 3. 各バージョンに対してOCRを実行
        if ocr_engine == 'easyocr':
            ocr_output = easyocr_reader.readtext(img_to_scan)
            for (bbox, text, prob) in ocr_output:
                top_left = bbox[0]
                width = int(bbox[1][0] - bbox[0][0])
                height = int(bbox[2][1] - bbox[1][1])
                all_results.append({
                    "text": text,
                    "box": (int(top_left[0] / scale_factor), int(top_left[1] / scale_factor), 
                            int(width / scale_factor), int(height / scale_factor)),
                    "confidence": prob
                })

        elif ocr_engine == 'tesseract':
            config = '--psm 11 --oem 3' # 参考コードの設定を利用
            ocr_output = pytesseract.image_to_data(img_to_scan, lang='jpn+eng', config=config, output_type=pytesseract.Output.DICT)
            for i in range(len(ocr_output['text'])):
                conf = int(ocr_output['conf'][i])
                text = ocr_output['text'][i].strip()
                if conf > 30 and text != '': # 参考コードの信頼度閾値を利用
                    all_results.append({
                        "text": text,
                        "box": (int(ocr_output['left'][i] / scale_factor), int(ocr_output['top'][i] / scale_factor),
                                int(ocr_output['width'][i] / scale_factor), int(ocr_output['height'][i] / scale_factor)),
                        "confidence": conf / 100.0
                    })
    
    # 4. 重複結果のフィルタリング（簡単のため、テキストと座標が完全に同じものを除く）
    unique_results = []
    seen = set()
    for res in all_results:
        # タプルはハッシュ可能なのでsetで使える
        identifier = (res['text'], res['box'])
        if identifier not in seen:
            unique_results.append(res)
            seen.add(identifier)
            
    return unique_results

def phase2_scan_with_ocr(image_path: str, region: Dict[str, int], ocr_engine: str = 'easyocr') -> List[Dict]:
    """Phase 2: 指定された領域をOCRでスキャンする（新しい高精度版）"""
    print(f"\n--- Phase 2: 高精度OCRスキャン ({ocr_engine.upper()}) ---")
    try:
        img = Image.open(image_path)
        crop_box = (region['x'], region['y'], region['x'] + region['width'], region['y'] + region['height'])
        crop_box = (max(0, crop_box[0]), max(0, crop_box[1]), min(img.width, crop_box[2]), min(img.height, crop_box[3]))
        
        cropped_img_pil = img.crop(crop_box)
        cropped_img_pil.save("debug_cropped_image.png")
        print("DEBUG: トリミングした画像を debug_cropped_image.png として保存しました。")

        # Pillow ImageをOpenCV (Numpy) 形式に変換
        cropped_img_cv = cv2.cvtColor(np.array(cropped_img_pil), cv2.COLOR_RGB2BGR)

        # 新しい高精度OCR関数を呼び出し
        results = advanced_ocr_scan(cropped_img_cv, ocr_engine)
        
        print(f"INFO: OCRで {len(results)}個のユニークなテキストを検出しました。")
        print(f"DEBUG: OCR結果: {results}")
        return results
    except Exception as e:
        print(f"ERROR: Phase 2でエラーが発生しました: {e}")
        return []

# ▲▲▲▲▲ OCR処理の刷新ここまで ▲▲▲▲▲


def phase3_verify_with_llm(cropped_image_path: str, ocr_results: List[Dict], target_text: str) -> Optional[List[str]]:
    # (この関数は変更なし)
    print("\n--- Phase 3: LLMによるOCR結果の検証 ---")
    if not ocr_results:
        print("INFO: OCR結果が空のため、検証をスキップします。")
        return None
    base64_image = encode_image_to_base64(cropped_image_path)
    ocr_candidates_str = "\n".join([f'- "{item["text"]}"' for item in ocr_results])
    prompt = f"""
    この画像を見てください。私は「{target_text}」というテキストを探しています。
    OCRでスキャンしたところ、以下のテキスト候補が見つかりました。
    --- OCR候補 ---
    {ocr_candidates_str}
    ---
    これらの候補の中から、私が探している「{target_text}」に意味的・視覚的に最も一致するものを特定してください。
    - 複数の候補が組み合わさって目的のテキストを形成している場合は、それら全てのテキストをリストで返してください。
    - 候補の中に完全に一致するものがある場合も、そのテキストをリストで返してください。
    - 誤認識されているが文脈的に正しいと思われるものがあれば、そのOCRが読み取ったテキストを返してください。
    あなたの回答は、以下のJSON形式でお願いします。
    {{ "found": true, "texts": ["text1", "text2", ...] }}
    または、該当するものがなければ
    {{ "found": false, "texts": [] }}
    他の説明は不要です。
    """
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"},},],}], max_tokens=200, response_format={"type": "json_object"})
        content = response.choices[0].message.content
        print(f"DEBUG: LLMからの応答: {content}")
        result = json.loads(content)
        if result.get("found"):
            print(f"INFO: LLMが一致するテキストを特定しました: {result['texts']}")
            return result["texts"]
        else:
            print("INFO: LLMは一致するテキストを見つけられませんでした。")
            return None
    except Exception as e:
        print(f"ERROR: Phase 3でエラーが発生しました: {e}")
        return None

# (Main Orchestratorは変更なし)
def find_text_coordinates(
    target_text: str,
    context_hint: str,
    ocr_engine: str = 'easyocr'
) -> Optional[Dict[str, int]]:
    screenshot_file = "debug_screenshot.png"
    take_screenshot(screenshot_file)
    img_size = Image.open(screenshot_file).size

    region = phase1_get_region_with_llm(screenshot_file, target_text, context_hint)
    if not region:
        print("最終結果: 失敗 (Phase 1で領域を特定できませんでした)")
        os.remove(screenshot_file)
        return None
        
    ocr_results = phase2_scan_with_ocr(screenshot_file, region, ocr_engine)
    
    print("\n--- 照合フェーズ ---")
    
    def find_target_in_ocr(results, original_region):
        for item in results:
            if item["text"] == target_text:
                print(f"INFO: 完全一致するテキストが見つかりました: '{target_text}'")
                abs_box = item["box"]
                return {"x": original_region['x'] + abs_box[0], "y": original_region['y'] + abs_box[1], "width": abs_box[2], "height": abs_box[3]}
        return None

    found_coords = find_target_in_ocr(ocr_results, region)
    if found_coords:
        print(f"最終結果: 成功！ 座標: {found_coords}")
        os.remove(screenshot_file)
        return found_coords

    print("\n--- フォールバック戦略発動 ---")
    print("INFO: LLMが提案した領域で完全一致が見つかりませんでした。より広い領域で再スキャンします。")
    
    fallback_region = {'x': 0, 'y': 0, 'width': img_size[0], 'height': img_size[1] // 4}
    print(f"INFO: フォールバック領域: {fallback_region}")

    ocr_results_fallback = phase2_scan_with_ocr(screenshot_file, fallback_region, ocr_engine)
    
    found_coords_fallback = find_target_in_ocr(ocr_results_fallback, fallback_region)
    if found_coords_fallback:
        print(f"最終結果: 成功！(フォールバック) 座標: {found_coords_fallback}")
        os.remove(screenshot_file)
        return found_coords_fallback
    
    print("\nINFO: 完全一致が見つからなかったため、LLMによる検証フェーズに移行します。")
    # 検証には、より多くの情報を含む可能性のあるフォールバック時のOCR結果を利用する
    llm_verified_texts = phase3_verify_with_llm("debug_cropped_image.png", ocr_results_fallback, target_text)
    
    if llm_verified_texts:
        target_bboxes = []
        # LLMが検証したテキストを、フォールバックOCR結果から探す
        ocr_search_pool = ocr_results_fallback
        for text in llm_verified_texts:
            found = False
            for item in ocr_search_pool:
                if item["text"] == text:
                    target_bboxes.append(item["box"])
                    ocr_search_pool.remove(item)
                    found = True
                    break
            if not found:
                print(f"WARNING: LLMが指定したテキスト '{text}' がOCR結果に見つかりませんでした。")

        if not target_bboxes:
            print("最終結果: 失敗 (LLMが特定したテキストがOCR結果に存在しませんでした)")
            os.remove(screenshot_file)
            return None

        combined_rel_box = get_combined_bbox(target_bboxes)
        final_coords = {"x": fallback_region['x'] + combined_rel_box[0], "y": fallback_region['y'] + combined_rel_box[1], "width": combined_rel_box[2], "height": combined_rel_box[3]}
        print(f"最終結果: 成功！ 座標: {final_coords}")
        os.remove(screenshot_file)
        return final_coords
    else:
        print("最終結果: 失敗 (LLMによる検証でもテキストを特定できませんでした)")
        os.remove(screenshot_file)
        return None