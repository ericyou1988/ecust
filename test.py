from selenium import webdriver
import time

def main():
    chrome_driver = 'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'  #chromedriver的文件位置
    b = webdriver.Chrome(executable_path = chrome_driver)
    b.get('https://www.google.com')
    time.sleep(5)
    b.quit()

if __name__ == '__main__':
    main()