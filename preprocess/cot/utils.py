import json
from typing import List, Dict
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import requests
import time

class ChatLLM():
    def __init__(self, base_url, key, prompt, model, temperature, max_tokens=None):
        self.base_url = base_url
        if type(key) is tuple:
            self.key = key[0]
        elif type(key) is str:
            self.key = key
        else:
            raise Exception('key must be a string or a tuple')
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def _async_chat(self, session, req: Dict):
        prompt = self.prompt.format(**req)
        # if req has image_Url
        images = req.get('images', None)
        content = [
            {"type": "text", "text": prompt}
        ]
        if images:
            for image in images:
                content.append(
                    {'type': 'image_url', 'image_url': f'data:image/jpeg;base64,{image}'}
                )
        data = json.dumps({
            'messages': [{'role': 'user', 'content': content}],
            'model': self.model,
            'stream': False,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        })
        
        for attempt in range(3):
            response_json = None
            try:
                async with session.post(
                    url=self.base_url,
                    headers={
                        'Authorization': f'Bearer {self.key}',
                        'Content-Type': 'application/json'
                        },
                    data=data
                ) as response:
                    response_json = await response.json()
                    return response_json['choices'][0]['message']['content']
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(30)
                else:
                    if response_json is None:
                        raise Exception(e)
                    else:
                        raise Exception(response_json)

    async def _async_chat_batch(self, reqs: List[Dict]):
        async with aiohttp.ClientSession() as session:
            tasks = [self._async_chat(session, request) for request in reqs]
            return await asyncio.gather(*tasks)

    def _sync_chat(self, req: Dict):
        """Synchronous chat for better stability"""
        prompt = self.prompt.format(**req)
        images = req.get('images', None)
        
        content = [{"type": "text", "text": prompt}]
        if images:
            for image in images:
                content.append({
                    'type': 'image_url', 
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{image}'
                    }
                })
        
        data = {
            'messages': [{'role': 'user', 'content': content}],
            'model': self.model,
            'stream': False,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        
        headers = {
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json'
        }
        
        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    return response_json['choices'][0]['message']['content']
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                if attempt < 2:
                    time.sleep(30)
                    print(f"Retry {attempt + 1}/3 after error: {e}")
                else:
                    raise e

    def chat_batch(self, requests: List[Dict]):
        """Process requests using ThreadPoolExecutor for parallelism"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._sync_chat, req) for req in requests]
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=180)  # 3 minute timeout per request
                    results.append(result)
                except Exception as e:
                    print(f"Request failed: {e}")
                    results.append(f"Error: {str(e)}")
            return results
    