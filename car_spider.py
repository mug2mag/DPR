import json
import re
import requests
from bs4 import BeautifulSoup
import math
import time
# 构造 Request headers
agent = 'ozilla/5.0 (Linux; Android 5.0; SM-G900P Build/LRX21T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Mobile Safari/537.36'
headers = {
    'User-Agent': agent
}


def get_categories(url):
    category_num = []
    categories = []
    web_data = requests.get(url, headers=headers)
    web_data.encoding = 'utf-8'
    soup = BeautifulSoup(web_data.text, 'lxml')
    hrefs = soup.select('#tab-hot-cont > ul > li > a')

    for href in hrefs:
        category = href.get_text()
        categories.append(category)
        print("正在提取的类别是：", category)
        url_herf = href.get('href')
        time_pattern = re.compile('(?<=forum-c-)\d+')
        num = time_pattern.search(url_herf).group(0) if time_pattern.search(url_herf) else None
        category_num.append(num)
    return category_num, categories


def get_total_items(url):
    web_data = requests.get(url, headers=headers)
    web_data.encoding = 'utf-8'
    soup = BeautifulSoup(web_data.text, 'lxml')
    jsonStr = soup.select('p')[0].get_text()
    total_items = json.loads(jsonStr)['result']['total']
    return total_items


def get_data_from_categories(url):
    questions = []
    web_data = requests.get(url, headers=headers)
    web_data.encoding = 'utf-8'
    soup = BeautifulSoup(web_data.text, 'lxml')
    jsonStr = soup.select('p')[0].get_text()
    contents = json.loads(jsonStr)['result']['items']
    for content in contents:
        print("title:", content['title'])
        questions.append(content['title'])
    return questions


if __name__ == '__main__':
    total_hook_url = 'https://club.autohome.com.cn/frontapi/data/page/club_get_topics_list?page_num=1&page_size=50&club_bbs_type=c&club_bbs_id={}&club_order_type=1'
    cate_page_url = 'https://club.autohome.com.cn/frontapi/data/page/club_get_topics_list?page_num={}&page_size=50&club_bbs_type=c&club_bbs_id={}&club_order_type=1'
    car_url = 'https://club.autohome.com.cn/'

    all_information = {}
    #获取汽车种类
    all_cate_num, cate_names = get_categories(car_url)
    for cate_num, cate_name in zip(all_cate_num, cate_names):
        cate_questions = []
        total_hook_url = total_hook_url.format(cate_num)
        total_items = int(get_total_items(total_hook_url))
        page_num = math.ceil(total_items/50)
        for i in range(1, page_num):
            cate_page_url = cate_page_url.format(i, cate_num)
            questions = get_data_from_categories(cate_page_url)
            cate_questions.append(questions)
            time.sleep(2)
        all_information[cate_name] = cate_questions

    with open('汽车之家.json', 'w') as f:
        json.dump(all_information, ensure_ascii=False, indent=2)




    # get_data_from_categories(url)
    # get_all_text()
