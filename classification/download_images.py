from bing_image_downloader import downloader
import os
import shutil

# 카테고리별 검색 키워드
data_config = {
    'payment': '결제',
    'study': '칠판 필기 노트',
    'shopping': '쇼핑 상품 옷',
    'travel': '여행 티켓 예약 맛집',
    'document': '카카오톡 대화 채팅 기사 뉴스',
    'non_info': '강아지 고양이 풍경 인물 음식'
}

print("=" * 60)
print("🖼️  이미지 다운로드 시작")
print("=" * 60)

# 임시 다운로드 폴더
temp_dir = 'temp_downloads'

for category, keyword in data_config.items():
    print(f"\n📥 다운로드 중: {category} ({keyword})...")
    
    # Bing에서 이미지 다운로드 (15장 - 불량 대비)
    downloader.download(
        keyword,
        limit=30,
        output_dir=temp_dir,
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True
    )
    
    # 다운로드된 폴더 찾기 (키워드 이름으로 생성됨)
    downloaded_folder = os.path.join(temp_dir, keyword)
    
    if os.path.exists(downloaded_folder):
        # train_data/카테고리명 폴더 생성
        target_folder = os.path.join('train_data', category)
        os.makedirs(target_folder, exist_ok=True)
        
        # 이미지 파일 이동 (10장만)
        files = [f for f in os.listdir(downloaded_folder) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for i, file in enumerate(files[:20]):  # 10장만
            src = os.path.join(downloaded_folder, file)
            dst = os.path.join(target_folder, f"{category}_{i+1:02d}.jpg")
            shutil.copy(src, dst)
            print(f"  ✅ {category}_{i+1:02d}.jpg")
        
        print(f"✅ {category}: {len(files[:20])}장 완료")
    else:
        print(f"⚠️ {category}: 다운로드 실패")

# 임시 폴더 삭제
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

print("\n" + "=" * 60)
print("✅ 다운로드 완료!")
print("=" * 60)
print("\n📁 train_data/ 폴더를 확인하세요.")