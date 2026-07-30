# 메모리 누수 방지를 위한 매 카테고리 종료 후 리소스 정리 (VRAM 내 세션 해제)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'[{category}] 리소스 정리 완료')

반드시! 메모리 누수 방지 코드 추가하기