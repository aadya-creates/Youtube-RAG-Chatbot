from youtube_transcript_api import YouTubeTranscriptApi

ytt_api=YouTubeTranscriptApi()
res=ytt_api.fetch("g1fq327snPU", languages=["en-US"])
print(res)