import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "https://hypophosphatasia.com"
FAQ_URL = BASE_URL + "/faqs"


def get_soup(url):
    page = requests.get(url)
    return BeautifulSoup(page.content, "html.parser")


if __name__ == "__main__":
    soup = get_soup(FAQ_URL)
    atags = soup.find_all("a", class_="faqs__list_item_link")
    qas = []
    for tag in atags:
        q = tag.text.strip()
        ans_link = BASE_URL + tag["href"]
        ans_soup = get_soup(ans_link)
        ans = ans_soup.find("div", class_="faq__content").text.strip()
        ans = " ".join(ans.split())
        qas.append({"question": q, "answer": ans, "source": ans_link})

    with open("hpp_qa.json", "w") as f:
        json.dump(qas, f, indent=4)

