# main.py

import time
from coordinate_finder import find_text_coordinates

def run_test_case(test_name: str, target_text: str, context_hint: str, ocr_engine: str):
    """テストケースを実行し、結果を表示する関数。"""
    print(f"\n{'='*20} テスト開始: {test_name} ({ocr_engine.upper()}) {'='*20}")
    print(f"ターゲット: '{target_text}'")
    print(f"ヒント: '{context_hint}'")
    
    # ユーザーが準備するための待機時間
    print("\n3秒後にスクリーンショットを撮影します。対象のアプリケーションを前面に表示してください...")
    time.sleep(3)
    
    start_time = time.time()
    
    coordinates = find_text_coordinates(
        target_text=target_text,
        context_hint=context_hint,
        ocr_engine=ocr_engine
    )
    
    end_time = time.time()
    
    print("\n--- テスト結果 ---")
    if coordinates:
        print(f"✅ 座標の特定に成功しました: {coordinates}")
    else:
        print("❌ 座標の特定に失敗しました。")
        
    print(f"処理時間: {end_time - start_time:.2f}秒")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # --- テストケース ---
    
    # ケース1: 簡単な例（メモ帳の「ファイル」メニュー）
    run_test_case(
        test_name="メモ帳のファイルメニュー",
        target_text="ファイル",
        context_hint="アプリケーションの左上にあるメニューバーの項目です。",
        ocr_engine='tesseract' # 'tesseract' に変更して試すことも可能
    )

    # ケース2: OCRが分割して認識しそうな例（VS Codeの「ターミナル」メニュー）
    # run_test_case(
    #     test_name="VS Codeの新しいターミナル",
    #     target_text="新しいターミナル",
    #     context_hint="画面上部のメニューバーにある「ターミナル」をクリックすると表示される項目です。",
    #     ocr_engine='easyocr'
    # )

    # ケース3: 専門用語の例（架空のアプリを想定）
    # このテストを実行する際は、例えばWebブラウザで 'Von Mises Stress' と検索した結果画面などを表示してください。
    # run_test_case(
    #     test_name="専門用語のテスト",
    #     target_text="Von Mises Stress",
    #     context_hint="解析結果を表示するウィンドウのタイトル部分、または凡例にあります。",
    #     ocr_engine='easyocr'
    # )
    
    # # Tesseractでも試してみる
    # run_test_case(
    #     test_name="専門用語のテスト (Tesseract)",
    #     target_text="Von Mises Stress",
    #     context_hint="解析結果を表示するウィンドウのタイトル部分、または凡例にあります。",
    #     ocr_engine='tesseract'
    # )