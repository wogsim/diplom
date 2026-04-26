import pandas as pd
from datetime import datetime
from tqdm import tqdm
import requests as rq
from typing import List, Dict, Any
from loguru import logger
import sys
import os
from telethon import TelegramClient
from typing import Any, Dict, List, Union, Optional

class VkWallGet:
    BASE_URL = "https://api.vk.com/method/wall.get"
    
    def __init__(self, access_token: str, api_version: str = "5.199"):
        self.access_token = access_token
        self.api_version = api_version

    def _one_request(self, domain: str, offset: int) -> Dict[str, Any]:
        params = {
            'domain': domain,
            'count': 100,
            'offset': offset,
            'access_token': self.access_token,
            'v': self.api_version
        }
        logger.info('Отправлен запрос с параметрами {params}')
        data = rq.get(self.BASE_URL, params=params).json()
        logger.info('Получен запрос с параметрами {params}')
        return data
    
    def req_wall(self,
                  domain: str,
                  total_posts: int | str = 'full',
                  dataframe: bool=True) -> List[Dict[str, Any]] | pd.DataFrame:
        
        logger.info('Инициализация группы')
        initial_request = self._one_request(domain, 0)
        total_available_posts = initial_request['response']['count']
        posts_to_fetch = (total_available_posts if total_posts == 'full'
                          else min(total_posts, total_available_posts))
        logger.info('Доступно постов {total_available_posts}')
        logger.info('Будет спаршенно {posts_to_fetch}')
        
        data = []
        for offset in tqdm(range(0, posts_to_fetch, 100), desc="Fetching posts"):
            response = self._one_request(domain, offset)
            try:
                for post in response['response']['items']:
                    date = datetime.fromtimestamp(post['date']).strftime('%Y-%m-%d %H:%M:%S')
                    dict_post = {'date': date,
                                'text': post['text'],
                                'likes_count': post['likes']['count'],
                                'reposts_count': post['reposts']['count'],
                                'comments_count': post['comments']['count'],
                            }
                    if 'views' in post:
                        dict_post['views_count'] = post['views']['count']
                    else:
                        dict_post['views_count'] = None
                    data.append(dict_post)
            except Exception as e:
                logger.exception('Произошла ошибка')
                
        if dataframe:
            return pd.DataFrame(data)
        return data

class TelegramChanelParser:
    def __init__(self, api_id: str, api_hash: str, logger=logger) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.logger = logger
        self.client: Optional[TelegramClient] = None

    async def connect_client(self, session_name: str = 'session_name') -> None:
        self.client = TelegramClient(session_name, self.api_id, self.api_hash)
        self.logger.info("Connecting to Telegram client...")
        await self.client.start()
        self.logger.info("Connected to Telegram client.")

    async def disconnect_client(self) -> None:
        if self.client and self.client.is_connected():
            await self.client.disconnect()
            self.logger.info("Disconnected from Telegram client.")

    async def fetch_posts(self, channel: str, dataframe: bool = True, **kwargs) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        if not self.client or not self.client.is_connected():
            raise RuntimeError("Telegram client is not connected. Call connect_client() first.")

        posts: List[Dict[str, Any]] = []
        self.logger.info(f"Fetching posts from channel: {channel}")

        async for message in self.client.iter_messages(channel, reverse=True, **kwargs):
            reactions = {}
            if message.reactions:
                for r in message.reactions.results:
                    key = getattr(getattr(r, "reaction", None), "emoticon", None) or str(getattr(r, "reaction", ""))
                    reactions[key] = r.count

            replies_count = message.replies.replies if message.replies else 0

            if message.text or message.media:
                posts.append({
                    'id': message.id,
                    'date': message.date.strftime('%Y-%m-%d %H:%M:%S') if message.date else None,
                    'text': message.text,
                    'views_count': message.views,
                    'forwards_count': message.forwards,
                    'replies_count': replies_count,
                    'reactions': reactions or None
                })

        self.logger.info(f"Fetched {len(posts)} posts from channel: {channel}")
        return pd.DataFrame(posts) if dataframe else posts