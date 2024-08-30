import os
import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# 상수
PIXELS_URL = "https://www.pexels.com/ko-kr/search/face/"
IMAGE_DIR = "./data/"

# 크롬 드라이버 경로 설정
chrome_options = Options()
#chrome_options.add_argument("--headless")  # 브라우저 창을 띄우지 않음
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")  # GPU 가속 비활성화
chrome_options.add_argument("window-size=1920x1080")  # 창 크기를 명시적으로 지정
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

# 크롬 드라이버 서비스 설정
service = Service("/usr/local/share/chromedriver")  # 여기서 '/path/to/chromedriver'를 실제 크롬 드라이버 경로로 변경해야 합니다.
driver = webdriver.Chrome(service=service, options=chrome_options)


def crawling_pixels(row, image_name, last_height):
    # 소스코드 갱신
    while True:
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")

        for column in range(1, 4):
            img_tag = soup.select("#\- > div.Grid_gridWrapper__FBUdI.BreakpointGrid_grid-spacing-mobile-15__uwBk_.BreakpointGrid_grid-spacing-tablet-15__cNW4v.BreakpointGrid_grid-spacing-desktop-20__YZDXE.BreakpointGrid_grid-spacing-oversized-20__xO0KB > div > div:nth-child(" + str(column) + ") > div:nth-child(" + str(row) + ") > article > a > img")

            if img_tag:
                img_src = img_tag[0].get('src')
                print(img_src)

                img_path = IMAGE_DIR + str(image_name) + '.jpg'
                download_image(img_src, img_path)

                image_name += 1


        from selenium.webdriver import ActionChains

        #ActionChains생성
        action = ActionChains(driver)

        #리스트 가져오기
        bottom = driver.find_elements(By.CSS_SELECTOR, "#\- > div.Grid_gridWrapper__FBUdI.BreakpointGrid_grid-spacing-mobile-15__uwBk_.BreakpointGrid_grid-spacing-tablet-15__cNW4v.BreakpointGrid_grid-spacing-desktop-20__YZDXE.BreakpointGrid_grid-spacing-oversized-20__xO0KB > div > div:nth-child(" + str(column) + ") > div:nth-child(" + str(row) + ") > article > a > img")

        #move_to_element를 이용하여 이동
        if bottom:
            action.move_to_element(bottom[0]).perform()


        # 약간 위로 스크롤
        driver.execute_script("window.scrollBy(0, -200);")
        time.sleep(3)

        row += 1


def download_image(image_url, save_path):
    # 이미지 요청
    response = requests.get(image_url)
    
    # 요청 성공 여부 확인
    if response.status_code == 200:
        # 이미지 데이터를 바이너리로 저장
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Image successfully downloaded: {save_path}")
    else:
        print(f"Failed to retrieve image. Status code: {response.status_code}")


if __name__ == '__main__':
    try:
        # 데이터 저장 디렉토리 생성
        if not os.path.exists(IMAGE_DIR):
            os.makedirs(IMAGE_DIR)

        # 웹 페이지 로드
        driver.get(PIXELS_URL)

        # 초기값
        row = 1
        image_name = 0
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        # 크롤링
        crawling_pixels(row, image_name, last_height)
    
    except Exception as e:
        print(e)
