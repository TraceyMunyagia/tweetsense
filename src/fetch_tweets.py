
import os, csv, random
from datetime import datetime, timedelta

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "raw_tweets.csv")
FIELDNAMES = ["id", "date", "content", "username", "likeCount", "retweetCount"]

POSITIVE = [
    "Really impressed by {t}! The capabilities are mind-blowing.",
    "{t} is genuinely changing how I work every single day.",
    "Just tried {t} and I'm blown away. The future is here!",
    "{t} solved a problem in 5 seconds that took me hours before.",
    "The progress in {t} over the last year is absolutely incredible.",
    "{t} keeps getting better. Kudos to the team behind it!",
    "Using {t} at work now and our output has doubled. Highly recommend.",
    "Hard to believe how good {t} has become. Truly remarkable.",
    "The {t} demo today was absolutely stunning. Game changer.",
    "{t} just helped me debug a nasty issue in minutes. Love it.",
    "Excited about what {t} means for the future of technology.",
    "{t} is the best productivity tool I have used in years.",
    "Incredible what {t} can do now. We are living in the future.",
    "{t} made my workflow so much faster. Cannot imagine going back.",
    "Loving every update to {t}. The team is crushing it.",
]
NEGATIVE = [
    "{t} still makes too many mistakes to rely on for serious work.",
    "Disappointed by the latest {t} update. Felt like a step backward.",
    "The hype around {t} is way overblown. It is just autocomplete.",
    "{t} hallucinated completely wrong info and I almost used it. Scary.",
    "Privacy concerns around {t} keep growing. Not comfortable using it.",
    "The cost of {t} APIs is getting ridiculous. Pricing us out.",
    "{t} is overhyped. Real intelligence is still far away.",
    "Frustrated with {t} today. Could not complete a simple task correctly.",
    "Worried about job displacement because of {t}. Real conversation needed.",
    "{t} gave me three different wrong answers. Back to Stack Overflow.",
    "Security risks in {t} are being ignored. This will bite us.",
    "{t} is inconsistent. Works great one day, terrible the next.",
    "Not impressed with {t} at all. Way too many errors for production.",
    "{t} keeps going down at the worst times. Reliability is a real issue.",
    "The bias in {t} outputs is a serious problem nobody wants to address.",
]
NEUTRAL = [
    "Reading about the latest {t} research. Interesting developments.",
    "{t} has pros and cons. Still evaluating whether it fits our stack.",
    "Attended a talk on {t} today. Lots to think about.",
    "The {t} landscape is changing fast. Hard to keep up.",
    "Not sure what to make of {t} yet. Need more time with it.",
    "{t} is being discussed a lot in our team. No consensus yet.",
    "Comparing {t} tools this week. Each has different trade-offs.",
    "{t} is useful for some tasks, not so much for others.",
    "New {t} paper dropped. Haven't read it yet but looks interesting.",
    "Curious where {t} will be in two years. Anyone have predictions?",
    "Our company is evaluating {t} for internal use. Early days still.",
    "{t} is interesting but I need more time to form a real opinion.",
    "Watched a demo of {t} today. Impressive in some areas, limited in others.",
    "Still learning how to get the best results out of {t}.",
    "The {t} community is growing fast. Lots of debate about best practices.",
]
TOPICS = [
    "ChatGPT","GPT-4","Claude","Gemini","GitHub Copilot","OpenAI",
    "machine learning","artificial intelligence","large language models",
    "generative AI","deep learning","Mistral","open source AI","LLMs","AI tools",
]
USERS = ["tech_tracy","devmike92","airesearcher","pythonista_ke","mlops_daily",
         "data_nerd254","nairobi_dev","ai_skeptic","techie_zuri","cto_musings"]

def generate(n=1000):
    pos_n = int(n*0.40); neg_n = int(n*0.25); neu_n = n-pos_n-neg_n
    def batch(tpl, count):
        return [random.choice(tpl).format(t=random.choice(TOPICS)) for _ in range(count)]
    contents = batch(POSITIVE,pos_n)+batch(NEGATIVE,neg_n)+batch(NEUTRAL,neu_n)
    random.shuffle(contents)
    base = datetime(2024,1,1)
    return [{"id":str(i+1),"date":(base+timedelta(hours=i*1.5)).isoformat(),
             "content":c,"username":random.choice(USERS),
             "likeCount":random.randint(0,800),"retweetCount":random.randint(0,200)}
            for i,c in enumerate(contents)]

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("[INFO] Generating 1000 synthetic tech tweets...")
    tweets = generate(1000)
    with open(OUTPUT_FILE,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader(); w.writerows(tweets)
    print(f"[INFO] Saved {len(tweets)} tweets → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
