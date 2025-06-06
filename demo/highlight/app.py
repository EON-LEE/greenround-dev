import os
import logging
from swing_highlight_generator import SwingHighlightGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # 입력 비디오 경로
        video_path = "path/to/your/swing/video.mp4"
        
        # 출력 디렉토리 생성
        output_dir = "highlights"
        os.makedirs(output_dir, exist_ok=True)
        
        # 출력 파일 경로
        output_path = os.path.join(output_dir, "swing_highlights.mp4")
        
        # 하이라이트 생성기 초기화
        generator = SwingHighlightGenerator()
        
        # 스윙 분석 수행
        logger.info("Analyzing swing...")
        analysis_result = generator.analyze_swing(video_path)
        
        if not analysis_result:
            logger.error("Failed to analyze swing")
            return
            
        # 하이라이트 영상 생성
        logger.info("Generating highlights...")
        success = generator.generate_highlights(video_path, output_path)
        
        if success:
            logger.info(f"Highlights generated successfully: {output_path}")
        else:
            logger.error("Failed to generate highlights")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 