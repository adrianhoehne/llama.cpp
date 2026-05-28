[Skip to main content](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/experts_first_llamacpp/#main-content)
 Experts first llama.cpp : r/LocalLLaMA

Open menu[](https://www.reddit.com/)

[Open App](https://reddit.app.link/?%24android_deeplink_path=reddit%2Fr%2FLocalLLaMA%2Fcomments%2F1tknbzh%2Fexperts_first_llamacpp%2F&%24deeplink_path=%2Fr%2FLocalLLaMA%2Fcomments%2F1tknbzh%2Fexperts_first_llamacpp%2F&%24og_redirect=https%3A%2F%2Fwww.reddit.com%2Fr%2FLocalLLaMA%2Fcomments%2F1tknbzh%2Fexperts_first_llamacpp%2F&base_url=%2Fr%2FLocalLLaMA%2Fcomments%2F1tknbzh%2Fexperts_first_llamacpp%2F&mweb_loid=t2_2ex13nxhqc&mweb_loid_created=1779467489130&referrer_domain=www.reddit.com&referrer_url=%2Fr%2FLocalLLaMA%2Fcomments%2F1tknbzh%2Fexperts_first_llamacpp%2F&campaign=no_amp_test&utm_name=no_amp_test&channel=xpromo&utm_source=xpromo&feature=mweb3x&utm_medium=mweb3x&keyword=no_amp&utm_term=no_amp&tags=top_nav_get_app&utm_content=top_nav_get_app)

[Go to Reddit Answers](https://www.reddit.com/answers/) Expand search Expand user menu

[![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=96&height=96&frame=1&auto=webp&crop=96%3A96%2Csmart&s=0ae51826e9b99ef23f985ae5b3c6c8d76b205c79)\
\
Go to LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)

[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)  ![Llama](https://emoji.redditmedia.com/23w2nhjj1e9f1_t5_81eyvm/Llama)28m ago

[comanderxv](https://www.reddit.com/user/comanderxv/)

Experts first llama.cpp
=======================

[Discussion](https://www.reddit.com/r/LocalLLaMA/?f=flair_name%3A%22Discussion%22)

This is for all with 12GB VRAM.

Hi, I created a fork of llama.cpp with an experimental implementation of experts instead of layers. The reason is I own an RTX 2060 with 12GB VRAM. That sounds big but is too little for dense models. That is why I use mainly MoE models because of that. The problem is, you need to split some layers to the CPU lane.

As you all surely know, Qwen3.6-35B-A3B uses only 8 experts per token; the rest are unused, so why not fill the experts into VRAM instead of complete layers full of unused experts?

I started to create a UI to monitor which experts are used. This already showed me that the first layers are more important to have on VRAM than the last ones; the reason is that they would change the experts more frequently than the others. Unfortunately, n-cpu-moe with llama.cpp will let the first layers on the CPU, so I tried -ot, but that's another story. With the optimized setup, I was able to reach about 22 tk/s. (Remember the 2060 has only about half the CUDA cores of a 3060.) With the default --n-cpu-moe, I get 19 tk/s

I only run Q6 models, since the degradation at coding is visible. My context is not quantized (same reason), and because of Java development, I need a big context window of 100k.

However, with my expert variant and a hit rate of about 62%, it increased to 26 tks. The break-even point was at a 42% hit rate. This means the prompt has used 42% of the chosen experts on the GPU in my cache. As I tested smaller sizes of RAM (built-in argument to specify the VRAM usage), another use case came into my mind. With a good profile, you can reduce the usage a lot without sacrificing speed.

Now, to my question. Is there a person who would like to give it a test? I really would like to know how it behaves on a 3060/4060 or similar. (CUDA is a requirement, and Qwen 35B A3B or Gemma 26B A4B). **Currently, it is tested only on Linux.**

Really, I don't want to earn any stars or so. I don't care; I just want to know how much it increases the token generation on which NVIDIA graphics card.

It would need the following: checkout and build [https://github.com/adrianhoehne/llama.cpp](https://github.com/adrianhoehne/llama.cpp)

Start it with the additional arguments:

./build/bin/llama-server --moe-layer-perf-out experts.json \\
--cpu-moe \\
--ctx-size 100000 \\
--parallel 1

Then start a prompt and wait. This will take longer than usual because every expert is still on the CPU.

After that, exchange the arguments to

./build/bin/llama-server --moe-hot-cache experts.json \\
--moe-hot-cache-max-mib -1 \\
--moe-hot-cache-auto-reserve-mib 1024 \\
--moe-hot-cache-update-rate 0.10 \\
--cpu-moe \\
--ctx-size 100000 \\
--parallel 1

And start measurement.

I also included the view of which experts are used to the Llama UI:

[![r/LocalLLaMA - Button for ui](https://preview.redd.it/experts-first-v0-1yy5050qgp2h1.png?width=238&format=png&auto=webp&s=3d9d57c39e3ca19122cea19487fac5d59556264c)](https://preview.redd.it/experts-first-v0-1yy5050qgp2h1.png?width=238&format=png&auto=webp&s=3d9d57c39e3ca19122cea19487fac5d59556264c "Image from r/LocalLLaMA - Button for ui")

Button for ui

Share [![u/LOTTO24_AG avatar](https://styles.redditmedia.com/t5_90ahhr/styles/profileIcon_avpbp9ze2tef1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=72c3e0dc72d1e979edc49e35db937ff387593d75)LOTTO24\_AG](https://www.reddit.com/user/LOTTO24_AG/) • [Promoted](https://www.reddit.com/user/LOTTO24_AG/)

120.000.000€ im Eurojackpot – was würdest Du tun?

![Thumbnail image: 120.000.000€ im Eurojackpot – was würdest Du tun?](https://preview.redd.it/k1l3zi0vmg2h1.jpeg?auto=webp&s=c1d97d734e0a539e8709e798bc2f207e90c4808b)

lotto24.de

Learn More

[](https://www.reddit.com/user/jacek2023/)

[jacek2023](https://www.reddit.com/user/jacek2023/)

• [14m ago](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/comment/on9ruc0/)

 ![Profile Badge for the Achievement Top 1% Poster](https://i.redd.it/gqujlodqi3yd1.png) Top 1% Poster

This is whole implementation of --n-cpu-moe

[![Comment Image](https://preview.redd.it/experts-first-v0-iv8ik0amrp2h1.png?width=2156&format=png&auto=webp&s=203c8cc6be4cdf484688cab4588cab265c3af64a)](https://preview.redd.it/experts-first-v0-iv8ik0amrp2h1.png?width=2156&format=png&auto=webp&s=203c8cc6be4cdf484688cab4588cab265c3af64a)

if I understand your idea correctly you just need to pick different layers instead of:

inline std::string llm\_ffn\_exps\_block\_regex(int idx) {
    return string\_format("blk\\\\.%d%s", idx, LLM\_FFN\_EXPS\_REGEX);
}

I am pasting this because I tried to open your code and I see million of lines doing something

[](https://www.reddit.com/user/comanderxv/)

[comanderxv](https://www.reddit.com/user/comanderxv/)

• [5m ago](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/comment/on9toy8/)

\--ncmoe puts the first complete layers to the cpu lane. So, yes you could optimize by choosing the right layer and set it up with -ot which I did in the beginning since I saw that the experts change a lot in the very first layer. Which makes sense. But that wasnt enough.

1 more reply

1 more reply

[1 more reply](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/comment/on9toy8/?force-legacy-sct=1) [More replies](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/comment/on9ruc0/?force-legacy-sct=1)

[![u/RemarkableAntelope80 avatar](https://www.redditstatic.com/avatars/defaults/v2/avatar_default_2.png)](https://www.reddit.com/user/RemarkableAntelope80/)

[RemarkableAntelope80](https://www.reddit.com/user/RemarkableAntelope80/)

• [18m ago](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/comment/on9qvtn/)

That is an awesome increase if true, and if it fits properly back into mainline. Great work.

It makes sense, various people had results that some experts were used a _lot_ more than others. That was in the context of pruning though, and the trouble with that is, rare activation doesn't mean unimportant. I think the experts tended to specialise on different kinds of grammar and language stuff, rather than knowledge/skill areas. So the thing forgot how to think, or how to stop, or some other rare but critical thing. This seems a much smarter way to exploit it. Just have VRAM forget the layer exists, until that 1 in a hundred time when it's important.

I'm also in the 12GB boat, obviously for us, squeezing it in means losing more than 1 in a hundred, but I guess that's still more efficient. Super cool.

[](https://www.reddit.com/user/comanderxv/)

[comanderxv](https://www.reddit.com/user/comanderxv/)

• [10m ago](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/comment/on9so2f/)

Thanks. The speed increase depends on the experts list and their hitrate. If you switch the topic then it obviously goes down., since the cache contains the wrong experts. But I also implement an update after each prompt so in best case the hitrate increases after each until the optimum is reached.

1 more reply

1 more reply

[1 more reply](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/comment/on9so2f/?force-legacy-sct=1) [More replies](https://www.reddit.com/r/LocalLLaMA/comments/1tknbzh/comment/on9qvtn/?force-legacy-sct=1)

![Llama](https://emoji.redditmedia.com/23w2nhjj1e9f1_t5_81eyvm/Llama)

Public

Anyone can view, post, and comment to this community

0 0

*   [OK I get it, now I love llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1q7uuxo/ok_i_get_it_now_i_love_llamacpp/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 4mo ago
    
    [### OK I get it, now I love llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1q7uuxo/ok_i_get_it_now_i_love_llamacpp/)
    
    242 upvotes · 59 comments
    
    * * *
    
*    [![u/hyundai_de avatar](https://styles.redditmedia.com/t5_f1jyi7/styles/profileIcon_aai69wjfanff1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=93240ca08d169e7f3326784a848f614079f944a5)hyundai\_de](https://www.reddit.com/user/hyundai_de/) • [Promoted](https://www.reddit.com/user/hyundai_de/)
    
    Bereit für ein Upgrade? Sportliches Design trifft hier auf erstklassige Konditionen. Dein neues Fahrgefühl wartet schon. Starte jetzt – mit dem Hyundai KONA N Line im attraktiven Power Leasing ab 179 EUR mtl¹. NEXT START NOW.
    
    ![Thumbnail image: Bereit für ein Upgrade? Sportliches Design trifft hier auf erstklassige Konditionen. Dein neues Fahrgefühl wartet schon. Starte jetzt – mit dem Hyundai KONA N Line im attraktiven Power Leasing ab 179 EUR mtl¹. NEXT START NOW.](https://external-preview.redd.it/tNHGSQwNV_c2UD86a798trCg-btHNZ9wMdCqDdsbOIA.jpg?width=108&crop=smart&auto=webp&s=22dcad7ffe165d57582bd6d9c6420222e251bcdc)
    
    hyundai.com
    
    Learn More
    
    * * *
    
*   [llama.cpp is a vibe-coded mess](https://www.reddit.com/r/LocalLLaMA/comments/1s7i5mj/llamacpp_is_a_vibecoded_mess/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 2mo ago
    
    [### llama.cpp is a vibe-coded mess](https://www.reddit.com/r/LocalLLaMA/comments/1s7i5mj/llamacpp_is_a_vibecoded_mess/)
    
    45 comments
    
    * * *
    
*   [Lads, time to recompile llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1rmoi8d/lads_time_to_recompile_llamacpp/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 3mo ago
    
    [### Lads, time to recompile llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1rmoi8d/lads_time_to_recompile_llamacpp/)
    
    112 upvotes · 52 comments
    
    * * *
    
*   [Llama.cpp now with a true reasoning budget!](https://www.reddit.com/r/LocalLLaMA/comments/1rr6wqb/llamacpp_now_with_a_true_reasoning_budget/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 2mo ago
    
    [### Llama.cpp now with a true reasoning budget!](https://www.reddit.com/r/LocalLLaMA/comments/1rr6wqb/llamacpp_now_with_a_true_reasoning_budget/)
    
    [![r/LocalLLaMA - Llama.cpp now with a true reasoning budget!](https://external-preview.redd.it/UoDMb6y9PnoNMGYzzfBddWAOyi0g7ECHnxy2mYZoQ7U.png?width=140&height=70&auto=webp&s=469114b3e536b67f75968bfc18ee32265912beba)](https://github.com/ggml-org/llama.cpp/commit/acb7c790698fa28a0fbfc0468804926815b94de3 "Link from r/LocalLLaMA - Llama.cpp now with a true reasoning budget!")
    
    github
    
    342 upvotes · 71 comments
    
    * * *
    
*   [Llama.cpp rpc experiment](https://www.reddit.com/r/LocalLLaMA/comments/1q9yd1w/llamacpp_rpc_experiment/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 4mo ago
    
    [### Llama.cpp rpc experiment](https://www.reddit.com/r/LocalLLaMA/comments/1q9yd1w/llamacpp_rpc_experiment/)
    
    6 upvotes · 31 comments
    
    * * *
    
*   [Will llama.cpp multislot improve speed?](https://www.reddit.com/r/LocalLLaMA/comments/1sw1d79/will_llamacpp_multislot_improve_speed/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 26d ago
    
    [### Will llama.cpp multislot improve speed?](https://www.reddit.com/r/LocalLLaMA/comments/1sw1d79/will_llamacpp_multislot_improve_speed/)
    
    9 upvotes · 18 comments
    
    * * *
    
*   [Switching from Ollama to llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1qv8ah3/switching_from_ollama_to_llamacpp/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 4mo ago
    
    [### Switching from Ollama to llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1qv8ah3/switching_from_ollama_to_llamacpp/)
    
    5 upvotes · 14 comments
    
    * * *
    
*   [What do you implement after Llama.cpp?](https://www.reddit.com/r/LocalLLaMA/comments/1s69a9h/what_do_you_implement_after_llamacpp/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 2mo ago
    
    [### What do you implement after Llama.cpp?](https://www.reddit.com/r/LocalLLaMA/comments/1s69a9h/what_do_you_implement_after_llamacpp/)
    
    12 upvotes · 20 comments
    
    * * *
    
*   [Llama.cpp quantization is broken](https://www.reddit.com/r/LocalLLaMA/comments/1t3boe0/llamacpp_quantization_is_broken/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 18d ago
    
    [### Llama.cpp quantization is broken](https://www.reddit.com/r/LocalLLaMA/comments/1t3boe0/llamacpp_quantization_is_broken/)
    
    [![r/LocalLLaMA - Llama.cpp quantization is broken](https://external-preview.redd.it/9lnXiEQiDqnlT_EwDrk19oRQrZlykt52rBzzuNIglWk.png?width=140&height=75&auto=webp&s=3446a5eb9cb503bb21c185cf65310cb0dbb8d716)](https://www.reddit.com/r/LocalLLaMA/comments/1t3boe0/llamacpp_quantization_is_broken/)
    
    53 comments
    
    * * *
    
*   [Ran some Llama.cpp RPC test to see if its worth it. And if 10Gbe needed.](https://www.reddit.com/r/LocalLLaMA/comments/1t9lbcm/ran_some_llamacpp_rpc_test_to_see_if_its_worth_it/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 12d ago
    
    [### Ran some Llama.cpp RPC test to see if its worth it. And if 10Gbe needed.](https://www.reddit.com/r/LocalLLaMA/comments/1t9lbcm/ran_some_llamacpp_rpc_test_to_see_if_its_worth_it/)
    
    [![r/LocalLLaMA - Ran some Llama.cpp RPC test to see if its worth it. And if 10Gbe needed.](https://preview.redd.it/96er85zewd0h1.png?width=140&height=42&auto=webp&s=3f8ee05334abd66c412358931d119ee41fa41bee)](https://www.reddit.com/r/LocalLLaMA/comments/1t9lbcm/ran_some_llamacpp_rpc_test_to_see_if_its_worth_it/)
    
    30 upvotes · 22 comments
    
    * * *
    
*   [llama.cpp and Qwen CPU Only](https://www.reddit.com/r/LocalLLaMA/comments/1rqd4ik/llamacpp_and_qwen_cpu_only/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 2mo ago
    
    [### llama.cpp and Qwen CPU Only](https://www.reddit.com/r/LocalLLaMA/comments/1rqd4ik/llamacpp_and_qwen_cpu_only/)
    
    13 upvotes · 40 comments
    
    * * *
    
*   [Llama.cpp It runs twice as fast as LMStudio and Ollama.](https://www.reddit.com/r/LocalLLM/comments/1rsiahf/llamacpp_it_runs_twice_as_fast_as_lmstudio_and/)
    
    [![](https://styles.redditmedia.com/t5_84a9er/styles/communityIcon_7wizpqj3o0xa1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=63d5879dfb767d352df7dc02f950daae48b225fd)\
    \
    r/LocalLLM](https://www.reddit.com/r/LocalLLM/)
    • 2mo ago
    
    [### Llama.cpp It runs twice as fast as LMStudio and Ollama.](https://www.reddit.com/r/LocalLLM/comments/1rsiahf/llamacpp_it_runs_twice_as_fast_as_lmstudio_and/)
    
    69 upvotes · 33 comments
    
    * * *
    
*   [Llama.cpp's auto fit works much better than I expected](https://www.reddit.com/r/LocalLLaMA/comments/1srvqar/llamacpps_auto_fit_works_much_better_than_i/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 1mo ago
    
    [### Llama.cpp's auto fit works much better than I expected](https://www.reddit.com/r/LocalLLaMA/comments/1srvqar/llamacpps_auto_fit_works_much_better_than_i/)
    
    146 upvotes · 73 comments
    
    * * *
    
*   [Llama.cpp is getting better with every update](https://www.reddit.com/r/LocalLLM/comments/1ta9n8k/llamacpp_is_getting_better_with_every_update/)
    
    [![](https://styles.redditmedia.com/t5_84a9er/styles/communityIcon_7wizpqj3o0xa1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=63d5879dfb767d352df7dc02f950daae48b225fd)\
    \
    r/LocalLLM](https://www.reddit.com/r/LocalLLM/)
    • 11d ago
    
    [### Llama.cpp is getting better with every update](https://www.reddit.com/r/LocalLLM/comments/1ta9n8k/llamacpp_is_getting_better_with_every_update/)
    
    138 upvotes · 65 comments
    
    * * *
    
*   [llama.cpp appreciation post](https://www.reddit.com/r/LocalLLaMA/comments/1psbx2q/llamacpp_appreciation_post/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 5mo ago
    
    [### llama.cpp appreciation post](https://www.reddit.com/r/LocalLLaMA/comments/1psbx2q/llamacpp_appreciation_post/)
    
    [![r/LocalLLaMA - llama.cpp appreciation post](https://b.thumbs.redditmedia.com/6U0W2XMEDzFuMauwyCStbD3_ihA5msPi4aQYke2-I6I.jpg)](https://www.reddit.com/r/LocalLLaMA/comments/1psbx2q/llamacpp_appreciation_post/)
    
    1.7K upvotes · 155 comments
    
    * * *
    
*   [What is llama.cpp or PC optimal settings?](https://www.reddit.com/r/LocalLLaMA/comments/1r5db7d/what_is_llamacpp_or_pc_optimal_settings/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 3mo ago
    
    [### What is llama.cpp or PC optimal settings?](https://www.reddit.com/r/LocalLLaMA/comments/1r5db7d/what_is_llamacpp_or_pc_optimal_settings/)
    
    5 upvotes · 14 comments
    
    * * *
    
*   [llama.cpp is the linux of llm](https://www.reddit.com/r/LocalLLaMA/comments/1srgpqp/llamacpp_is_the_linux_of_llm/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 1mo ago
    
    [### llama.cpp is the linux of llm](https://www.reddit.com/r/LocalLLaMA/comments/1srgpqp/llamacpp_is_the_linux_of_llm/)
    
    185 upvotes · 93 comments
    
    * * *
    
*   [Why doesn't any OSS tool treat llama.cpp as a first class citizen?](https://www.reddit.com/r/LocalLLaMA/comments/1sr140o/why_doesnt_any_oss_tool_treat_llamacpp_as_a_first/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 1mo ago
    
    [### Why doesn't any OSS tool treat llama.cpp as a first class citizen?](https://www.reddit.com/r/LocalLLaMA/comments/1sr140o/why_doesnt_any_oss_tool_treat_llamacpp_as_a_first/)
    
    311 upvotes · 114 comments
    
    * * *
    
*   [llama.cpp speculative checkpointing was merged](https://www.reddit.com/r/LocalLLaMA/comments/1sprdm8/llamacpp_speculative_checkpointing_was_merged/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 1mo ago
    
    [### llama.cpp speculative checkpointing was merged](https://www.reddit.com/r/LocalLLaMA/comments/1sprdm8/llamacpp_speculative_checkpointing_was_merged/)
    
    274 upvotes · 78 comments
    
    * * *
    
*   [Happy birthday, llama.cpp!](https://www.reddit.com/r/LocalLLaMA/comments/1rpxkw9/happy_birthday_llamacpp/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 2mo ago
    
    [### Happy birthday, llama.cpp!](https://www.reddit.com/r/LocalLLaMA/comments/1rpxkw9/happy_birthday_llamacpp/)
    
    [![r/LocalLLaMA - Happy birthday, llama.cpp!](https://external-preview.redd.it/I0MwmmrMgAO_AEZrzGNtx5dTFHuZhmfXfQph0x06dkw.png?width=140&height=70&auto=webp&s=207c42b05c5bac80389d3b0209931fc3552e0625)](https://github.com/ggml-org/llama.cpp/commit/26c084662903ddaca19bef982831bfb0856e8257 "Link from r/LocalLLaMA - Happy birthday, llama.cpp!")
    
    github
    
    312 upvotes · 17 comments
    
    * * *
    
*   [Just built an app using llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1qg9uvi/just_built_an_app_using_llamacpp/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 4mo ago
    
    [### Just built an app using llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1qg9uvi/just_built_an_app_using_llamacpp/)
    
    [![r/LocalLLaMA - Just built an app using llama.cpp](https://external-preview.redd.it/Z3NhMDNlYW5hNGVnMUNUtL3ojV8mRZI5MC7PHWnku738D1U5XxnZmpt8UVdx.png?width=140&height=140&crop=1%3A1%2Csmart&format=jpg&auto=webp&s=985e6121b0c9bb0bd9a4766add51a40aa3deaec7)](https://www.reddit.com/r/LocalLLaMA/comments/1qg9uvi/just_built_an_app_using_llamacpp/)
    
    0:59
    
    7 upvotes · 10 comments
    
    * * *
    
*   [How long for llama.cpp official support of MTP?](https://www.reddit.com/r/LocalLLaMA/comments/1t7ur1f/how_long_for_llamacpp_official_support_of_mtp/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 13d ago
    
    [### How long for llama.cpp official support of MTP?](https://www.reddit.com/r/LocalLLaMA/comments/1t7ur1f/how_long_for_llamacpp_official_support_of_mtp/)
    
    91 upvotes · 50 comments
    
    * * *
    
*   [huge improvement after moving from ollama to llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1sj6zz8/huge_improvement_after_moving_from_ollama_to/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 1mo ago
    
    [### huge improvement after moving from ollama to llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1sj6zz8/huge_improvement_after_moving_from_ollama_to/)
    
    125 upvotes · 74 comments
    
    * * *
    
*   [A Llama.cpp GUI for easy config and launch. Been working on it for a while! I hope this helps anyone looking for an easier way to play around with llama.cpp](https://www.reddit.com/r/LocalLLM/comments/1tf08xv/a_llamacpp_gui_for_easy_config_and_launch_been/)
    
    [![](https://styles.redditmedia.com/t5_84a9er/styles/communityIcon_7wizpqj3o0xa1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=63d5879dfb767d352df7dc02f950daae48b225fd)\
    \
    r/LocalLLM](https://www.reddit.com/r/LocalLLM/)
    • 6d ago
    
    [### A Llama.cpp GUI for easy config and launch. Been working on it for a while! I hope this helps anyone looking for an easier way to play around with llama.cpp](https://www.reddit.com/r/LocalLLM/comments/1tf08xv/a_llamacpp_gui_for_easy_config_and_launch_been/)
    
    [![r/LocalLLM - A Llama.cpp GUI for easy config and launch. Been working on it for a while! I hope this helps anyone looking for an easier way to play around with llama.cpp](https://external-preview.redd.it/fKZOCbxYSa5S8S6w4YT-uzzXYKeF3nWwAsv8j375z2A.png?width=140&height=70&auto=webp&s=372b38ba33b17087060579d87ca5a4f9e9e25770)](https://github.com/thomas9120/LLama-GUI "Link from r/LocalLLM - A Llama.cpp GUI for easy config and launch. Been working on it for a while! I hope this helps anyone looking for an easier way to play around with llama.cpp")
    
    github
    
    27 upvotes · 7 comments
    
    * * *
    
*   [Llama.cpp server running ~2 weeks straight. Loses its mind?](https://www.reddit.com/r/LocalLLaMA/comments/1tdffl1/llamacpp_server_running_2_weeks_straight_loses/)
    
    [![](https://styles.redditmedia.com/t5_81eyvm/styles/communityIcon_cumnsvx9kzma1.png?width=48&height=48&frame=1&auto=webp&crop=48%3A48%2Csmart&s=a65b055886461d9f520fb038a7ab11356a72b896)\
    \
    r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
    • 8d ago
    
    [### Llama.cpp server running ~2 weeks straight. Loses its mind?](https://www.reddit.com/r/LocalLLaMA/comments/1tdffl1/llamacpp_server_running_2_weeks_straight_loses/)
    
    25 comments
    
    * * *