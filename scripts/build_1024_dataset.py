import json
import random
import re
from pathlib import Path

# Load existing real harvested notes
existing_notes = []
seen_titles = set()

for path in ["data/train_140.json", "data/val_61.json", "data/xhs_200_feed_notes.json", "data/edge_harvested_new.json"]:
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            items = json.load(f)
            for it in items:
                title = it.get("title", "").strip()
                if title and len(title) > 2 and title not in seen_titles:
                    seen_titles.add(title)
                    existing_notes.append({
                        "title": title,
                        "snippet": it.get("snippet", "").strip() or None,
                        "author": it.get("author", "").strip() or None
                    })

print(f"Loaded {len(existing_notes)} unique existing real harvested notes.")

# Category Lexicon & Generator templates
CATEGORIES = [
    "科技数码", "萌宠动物", "美妆护肤", "穿搭时尚", "美食探店", "职场成长",
    "家居生活", "运动健康", "母婴育儿", "旅游出行", "情感心理", "摄影剪辑"
]

CATEGORY_CORPUS = {
    "科技数码": [
        ("M4 Pro Mac mini 终于到手！开发与大模型微调深度实测", "用了三天Xcode编译和MLX大模型跑分，功耗和散热表现令人惊叹", False),
        ("Claude 3.7 Sonnet 混合思考模式初体验，程序员彻底不用写代码了？", "实测高难度算法推导与Metal GPU Kernel生成，逻辑缜密到可怕", True),
        ("固态硬盘4KB随机读写为什么这么重要？普通用户避坑指南", "不要只看标称7000MB/s的顺序读取，日常软件启动和碎片加载全靠4K性能", False),
        ("苹果再一次跳号！iOS 20 提前泄露，全面转向本地AI交互", "控制中心与Siri深度集成本地端侧小模型，界面交互重构", True),
        ("老电脑拯救计划：Linux Mint体验报告，流畅度远超预期", "给十年前的ThinkPad刷入轻量发行版，日常码字与轻度开发完全不卡", False),
        ("机械键盘轴体终极科普：线性轴 vs 段落轴深度横评", "拆解了十几种客制化热门轴体，聊聊弹簧压力克数与触底手感", False),
        ("本地NAS搭建私有云盘与影音库全记录", "Docker部署Jellyfin与qBittorrent，内网穿透与外网访问完全搞定", False),
        ("RTX 5090 发布前瞻：显存位宽与功耗参数曝光", "架构升级带来AI算力飞跃，但散热装甲体积再次刷新认知", False),
        ("深扒所谓开源AI智能体：套壳API还是真自主思考？", "抓包分析多个热门Agent项目，底层提示词链条与工具调用逻辑全解析", False),
        ("折叠屏手机用了一年后的真实感受，劝退还是真香？", "折痕变化、铰链阻尼以及外屏使用频率的深度体验总结", True),
    ],
    "萌宠动物": [
        ("小猫咪能有什么坏心思呢？只是想霸占你的键盘罢了", "每次打开电脑写代码，它就一定要趴在空格键上打呼噜", False),
        ("捡到一只流浪玄猫，带去医院体检做驱虫的全流程记录", "医生说身体很健康，就是稍微有点营养不良，现在乖乖干饭", False),
        ("千万别养边牧！因为你会发现智商被狗子碾压了", "会自己开门、会分类玩具，甚至还会假装听不懂指令偷吃零食", True),
        ("新手养布偶猫必备用品清单与避坑指南", "很多网红玩具根本不需要买，猫粮成分表与化毛膏挑选技巧", False),
        ("猫咪绝育术后护理全攻略，看着心疼但真的恢复很快", "伊丽莎白圈选择、伤口观察以及术后饮食过渡指南", False),
        ("柯基的电臀到底有多治愈？周末草地放飞实拍", "短腿小柯基在草地上快乐狂奔，每一帧都值得做成表情包", False),
        ("流浪动物救助站一日义工体验：每一双眼睛都渴望被爱", "记录基地里等待领养的毛孩子们，领养代替购买真的很重要", False),
        ("爬宠入坑半年记录：守宫饲养箱造景与喂食日常", "温湿度控制、垫材挑选以及蜕皮期的注意事项汇总", False),
        ("小狗第一次见到雪地会怎样？这反应把我笑翻了", "一脚踩进去直接懵圈，然后开始满地打滚疯跑", False),
        ("猫咪掉毛季生存指南：几款吸毛神器的真实横评", "静电除毛刷、无线吸尘器与宠物空气净化器的实测对比", False),
    ],
    "美妆护肤": [
        ("早C晚A护肤公式到底怎么搭？敏感肌耐受建立全流程", "从低浓度温和过渡，抗老提亮不翻车的护肤心得", False),
        ("混油皮本命粉底液测评！持妆12小时不斑驳暗沉？", "大夏天实测防水控油效果，鼻翼两侧出油后的自然奶油肌质感", True),
        ("千万别买这些网红护肤品！成分党扒皮智商税清单", "概念性添加和重度香精防腐，扒一扒成分表里的猫腻", True),
        ("干皮救星！换季沙漠大干皮也能一夜回春的涂抹面膜", "厚敷二十分钟洗掉，皮肤肉眼可见地水润软嫩，卡粉彻底拜拜", False),
        ("新手零失误消肿眼妆教程：大地色系万能眼影画法", "单眼皮肿眼泡也能轻松放大双眼，扁平脸秒变深邃立体", False),
        ("专柜粉底 vs 平价粉底盲测！肉眼真的看得出区别吗？", "左右脸各上一半，经过8小时日常通勤带妆后的高清微距对比", False),
        ("刷酸去黑头闭口保姆级避坑：水杨酸与果酸的正确用法", "千万不要盲目高浓度上脸，建立耐受与补水修复才是核心", False),
        ("精简护肤一个月，皮肤状态反而比以前更稳定了", "停掉复杂的精华叠涂，只保留基础清洁、保湿与硬防晒", False),
        ("冷白皮显白唇釉合集：黄皮涂也毫不违和的绝美色号", "薄涂日常素颜提气色，厚涂气场全开的高级雾面质感", False),
        ("换季屏障受损泛红自救指南：B5与神经酰胺修护面霜", "退红舒缓效果实测，温和无刺激的换季安心必备单品", False),
    ],
    "穿搭时尚": [
        ("胶囊衣橱实践：用这7件基础单品搭出两周不重样OOTD", "白衬衫、直筒牛仔裤、风衣与乐福鞋的高级极简搭配法", False),
        ("小个子显高穿搭法则：158穿出168气场的比例优化技巧", "高腰线打造、裤长黄金比例以及鞋子配色的视觉延伸感", False),
        ("老钱风美学到底穿的是什么？面料与剪裁质感深度剖析", "拒绝大Logo堆砌，天然棉麻、羊绒羊毛与克制色系的魅力", False),
        ("梨形身材必看！遮假胯宽直筒裤与A字半身裙精选", "彻底告别显胯紧身裤，修饰腿型笔直修长的显瘦穿搭模板", False),
        ("秋季Cleanfit风格搭配灵感：松弛感日常通勤示范", "低饱和度大地色系组合，既有职场干练又不失舒适惬意", False),
        ("如何穿出少年感？中性风工装裤与基础款卫衣穿搭指南", "版型挑选是灵魂，oversize叠穿的层次感与鞋帽点缀", False),
        ("百元平价穿出千元高级感：普通人也能学会的配饰魔法", "一条丝巾、一块腕表与金属耳环如何瞬间点亮基础款穿搭", False),
        ("大骨架圆身女孩的显瘦穿搭心法：大领口与垂坠感面料", "V领打底衫、微垫肩西装外套与阔腿垂感西裤的避雷组合", False),
        ("户外山系机能风穿搭：兼具防风防雨与都市潮流的Gorpcore", "始祖鸟、巴塔哥尼亚与萨洛蒙的硬核混搭日常解析", False),
        ("黑白灰极简搭配如何摆脱沉闷无聊？面料质感碰撞解析", "皮质夹克与针织软糯内搭的视觉反差，利落有型不撞衫", False),
    ],
    "美食探店": [
        ("藏在老居民楼下的地道菌子火锅！鲜到眉毛掉下来的云南风味", "鸡汤底熬足四小时，见手青、牛肝菌煮够20分钟才能开动", False),
        ("街头盲盒探店：开了二十年的破烂小摊，味道竟然惊艳了", "老板坚持手工现做，排队半小时吃到的热腾腾猪油拌粉", True),
        ("私房烘焙日常：免揉欧包新手教程，外脆内软大气孔秘诀", "铸铁锅烘烤技巧与高水合面团发酵控制，厨房小白也能搞定", False),
        ("千万别来这家排队两小时的网红咖啡馆！踩雷大吐槽", "环境确实出片，但拿铁咖啡豆明显过萃焦苦，甜品甜到发齁", True),
        ("深夜食堂治愈时刻：一人食日式肥牛饭超简单复刻做法", "洋葱炒出焦糖色，调一碗灵魂日式照烧汁，五分钟搞定晚餐", False),
        ("菜市场寻宝：十块钱在早市能买到多少快乐的美食？", "刚出锅的现炸油条、手工热豆花和刚摘下的清甜水果", False),
        ("周末brunch自制指南：流心班尼迪克蛋与荷兰酱制作", "水波蛋完美定型小窍门，配上自烤酸面包与烟熏三文鱼", False),
        ("顺德寻味记：一天吃满七顿的吃货旅行终极攻略", "双皮奶、鱼生、煲仔饭与毋米粥火锅，胃容量彻底告急", False),
        ("减脂期也能大口吃的低卡快手晚餐：彩椒黑椒牛肉粒", "少油快炒保持鲜嫩多汁，低热量高蛋白饱腹感拉满", False),
        ("传统手工老红糖姜茶慢熬全过程：暖胃驱寒的冬日饮品", "小黄姜拍碎与古法红糖慢炖收汁，厨房弥漫浓郁辛甜香气", False),
    ],
    "职场成长": [
        ("大厂离职一年后，我从焦虑内耗到真正找回生活掌控感", "告别无休止的汇报PPT与虚无指标，聊聊自由职业的真实收支", False),
        ("如何向上管理？普通员工必学的高情商工作汇报法", "带方案请示领导、结构化结论先行以及定期对齐预期的小技巧", False),
        ("述职答辩高分PPT逻辑框架：让领导一眼看懂你的核心价值", "STAR法则量化成果，业务闭环思考与来年规划策略梳理", False),
        ("考公上岸后的真实体制内日常：并没有传说中那么躺平", "基层材料撰写、活动组织与公文规范处理的一线体验分享", False),
        ("跳槽薪资谈判技巧：如何优雅地争取到30%以上的涨薪？", "知己知彼的市场对标调研、底牌保留与HR周旋沟通话术", False),
        ("职场新人避坑：千万别把同事当朋友的五条扎心真相", "保持专业边界感、避免情绪化发泄与做好工作留痕的重要性", True),
        ("拒绝职场精神内耗：手把手教你建立钝感力与心理防护网", "把工作当成一场游戏副本，课题分离与下班彻底断联习惯", False),
        ("英语零基础到流畅外企商务沟通：我的高效口语学习法", "影子跟读训练法、场景模拟与商务邮件高频句式积累", False),
        ("程序员35岁危机真的存在吗？聊聊技术架构转型与副业探索", "技术深度与业务广度的平衡，构建个人技术影响力的思考", False),
        ("时间管理四象限法则：每天准时下班的高效工作流程", "告别伪忙碌，用批处理与时间块专注解决最重要紧急的事务", False),
    ],
    "家居生活": [
        ("老破小改造逆袭！25平米开间变身治愈系原木风小家", "拆除非承重墙打通采光，全屋定制储物柜让收纳翻倍", False),
        ("TOTO智能马桶盖与整体浴室选购避坑指南", "水压要求、即热式与储热式区别，喷嘴自洁功能实测", False),
        ("极简生活两年，扔掉这50样东西后家里变得无比清爽", "断舍离的不是物品而是执念，家务时间缩短三分之二", False),
        ("租房族必看！不伤墙面的高颜值软装改造好物清单", "魔术贴画框、落地氛围灯与可水洗亚麻地毯的搭配技巧", False),
        ("阳台微花园打造日记：朝北阳台也能爆花的绿植品种推荐", "常春藤、蕨类与矾根的耐阴植物组合，四季常青充满生机", False),
        ("全屋智能家居联动实测：离家自动关灯关空调真的太爽了", "Aqara人体传感器与智能窗帘电机设置，起床睡眠场景自动化", False),
        ("衣柜收纳终极解决方案：告别叠衣服的悬挂式收纳法", "统一衣架规格、按季节色彩分类，找衣服只需三秒钟", False),
        ("开放式厨房一年后使用感受：油烟真的会飘得满客厅都是吗？", "大吸力集成灶实测与台面清洁神器的日常维护总结", False),
        ("卧室氛围感天花板：无主灯设计与3000K色温的魔力", "见光不见灯的洗墙灯带布局，营造助眠放松的温馨空间", False),
        ("厨房小家电红黑榜：用过十几次后彻底闲置的智商税", "空气炸锅真香但面条机吃灰，买之前先看自己的做饭习惯", False),
    ],
    "运动健康": [
        ("跑步新手如何无痛跑完人生第一个5公里？配速与呼吸法", "小步幅高步频保护膝盖，腹式呼吸与跑前动态热身指南", False),
        ("帕梅拉魔鬼虐腹跟练一个月打卡！马甲线肉眼可见显现", "核心发力技巧与动作细节纠错，不伤腰颈椎的发力要领", False),
        ("羽毛球拍挑选避坑指南：4U还是3U？中杆软硬如何选择？", "糖水拍推荐与拉线磅数知识，新手进阶选拍干货全解析", False),
        ("普拉提与瑜伽到底有什么区别？亲身体验半年的真实对比", "深层小肌肉群控制与核心稳定强化，体态改善效果深度测评", False),
        ("减脂期必备高蛋白低卡食材库清单与快手食谱推荐", "鸡胸肉不柴的腌制方法，虾仁、牛肉与魔芋豆腐的花样吃法", False),
        ("健身房新手器械训练全流程：练胸练背黄金动作规范", "器械轨迹安全保护与离心收缩感受，告别盲目瞎练", False),
        ("滑雪季入门装备避坑：雪服雪镜与护具哪些该买哪些该租？", "雪鞋贴合度才是最重要的，护臀与护膝的安全认证标准", False),
        ("改善骨盆前倾与假胯宽：每天十分钟睡前矫正训练动作", "臀桥、死虫式与髋屈肌拉伸，让体态更加挺拔舒展", False),
        ("网球初学者正手击球动作精讲：转体发力与击球点判断", "告别纯靠手臂挥拍，下盘蹬地借力与随挥轨迹动作分解", False),
        ("运动手环与心率手表测评：运动心率监测到底有多准？", "有氧燃脂区间计算与VO2max最大摄氧量指标的科学解读", False),
    ],
    "母婴育儿": [
        ("宝宝辅食添加第一步：强化铁米粉与根茎类泥糊制作", "单一不过敏原则，辅食研磨工具与冷冻分装盒使用指南", False),
        ("新手爸妈必备婴儿推车选购横评：高景观 vs 轻便伞车", "减震避震性能、一键收车与座椅双向旋转功能全方位对比", False),
        ("婴儿睡眠训练实操记录：告别抱睡奶睡，宝宝学会自主入睡", "建立吃玩睡程序、睡前仪式感培养与哭声排查技巧", False),
        ("幼小衔接到底要不要提前学拼音算术？小学班主任真心话", "比起学科知识超前抢跑，专注力与生活自理能力才是关键", False),
        ("产后腹直肌分离与盆底肌修复经验分享：科学恢复体态", "凯格尔运动正确呼吸与腹部发力，避开盲目卷腹误区", False),
        ("儿童绘本分龄精选书单：让孩子爱上阅读的经典好书推荐", "0-3岁触摸洞洞书，3-6岁情绪管理与科普绘本深度推荐", False),
        ("宝宝发烧物理降温与退烧药使用常识：对乙酰氨基酚与布洛芬", "不同年龄段用药间隔与计量计算，何时必须去医院急诊", False),
        ("母乳喂养避坑全攻略：开奶排空、小白点堵奶自救手法", "吸奶器档位调节技巧与背奶妈妈母乳储存保鲜规范", False),
        ("如何应对两岁宝宝的Terrible Two逆反期？高情商育儿沟通", "接纳孩子的情绪崩溃，给有限的选择而不是命令对抗", False),
        ("安全座椅选购硬指标：i-Size欧洲安全认证与ISOFIX接口", "反向安装乘坐到四岁的重要性，透气面料与防侧撞气囊解析", False),
    ],
    "旅游出行": [
        ("川西自驾大环线7天6晚保姆级攻略：避坑路线与机位全解", "高反预防药物准备、折多山与理塘路况以及早晚防寒穿搭", False),
        ("新西兰南岛秋季自驾全景记录：在特卡波湖仰望纯净星空", "房车营地预订技巧、国际驾照公证与徒步路线避坑合集", False),
        ("麦理浩径全段徒步避坑攻略，千万别轻敌！风景绝美但挑战极大", "二段沙滩暴晒无补给点，带足电解质水与登山杖是保命关键", True),
        ("瑞士火车通票Swiss Travel Pass最省钱玩法攻略", "景观列车订座技巧、免费登山缆车路线与行李托运指南", False),
        ("穷游日本东京关东周游券JR Pass利用最大化指南", "富士山河口湖、轻井泽与日光日光社寺的高性价比行程规划", False),
        ("出国旅行千万别开国际漫游大坑！目的地本地eSIM卡实测", "机场取卡与线上扫码激活速度对比，网速稳定资费省一半", True),
        ("独自一人出国旅行安全指南：女性Solo Travel实用防身技巧", "贵重物品分装、假装有同伴话术与紧急求助联系方式备份", False),
        ("新疆独库公路通车自驾全纪实：一天经历四季的绝美风景", "山顶积雪与山脚花海，加油站分布与车辆动力检查注意事项", False),
        ("带爸妈去北京旅游怎么安排不累？四天三晚长辈友好路线", "故宫轮椅预约、无障碍通道与地道老字号不排队就餐攻略", False),
        ("海岛度假行李打包极简清单：手提行李箱搞定一周旅行", "速干衣物、潜水装备与便携防晒喷雾的过安检收纳诀窍", False),
    ],
    "情感心理": [
        ("深度内耗的本质是不敢做选择：停止精神内讧的四步自愈法", "学会接纳不完美的开局，行动是治愈一切焦虑的唯一解药", False),
        ("为什么越长大越喜欢独处？高质量独处带给人的心智重建", "在没有社交面具的安静时光里，重新倾听内心真实的声音", False),
        ("如何识别身边的隐性自恋者？恋爱与职场中的PUA特征解析", "煤气灯效应辨识、打压控制话术与果断切断联系的心理建设", False),
        ("讨好型人格如何勇敢说不？摆脱他人眼光评价的底层逻辑", "把注意力收回到自己身上，你的界限感决定了别人对待你的方式", False),
        ("亲密关系里的假性亲密：同居三年为什么依然感觉陌生遥远？", "缺乏脆弱情绪的深度袒露，如何打破情感隔离走向真正共鸣", False),
        ("社恐人的生存哲学：不迎合任何圈子，世界反而变得简单开阔", "低能量人格的充电机制，拒绝无效社交让生活更专注纯粹", False),
        ("经历重大挫折后如何重建生活？心理弹性（Resilience）培养", "允许自己短暂沉沦悲伤，把危机当成人格二次重塑的契机", False),
        ("焦虑发作时的五感着陆法：三分钟快速平复心跳与恐慌情绪", "调动视听触嗅味觉感受当下，把大脑从假想灾难中拉回现实", False),
        ("我们为什么总是爱上伤害自己的人？童年依恋模式对择偶的影响", "强迫性重复心理机制分析，识别原生家庭烙印打破恶性循环", False),
        ("原谅自己比原谅别人困难得多：学会与过去的遗憾达成和解", "当时的你已经做出了认知范围内的最优解，允许过去翻篇", False),
    ],
    "摄影剪辑": [
        ("富士相机胶片模拟配方调色参数分享：经典Classic Chrome", "高光色调-1阴影色调-2，色彩效果强，直出日系复古街道质感", False),
        ("索尼微单S-Log3曝光与还原LUT全流程保姆级教程", "向右曝光ETTR技巧，色彩空间色彩深度设置避免噪点断层", False),
        ("无人机航拍运镜黄金八法：新手也能飞出电影级航拍大片", "刷锅环绕、扣拍俯视与穿云前进镜头节奏控制要领", False),
        ("手机拍照构图黄金法则：九宫格与引导线如何让画面高级", "低角度仰拍拉长视觉纵深，前景虚化打造电影感层次", False),
        ("藏服写真旅拍早晚光线机位全解：香格里拉独克宗古城实拍", "转经筒转角与白塔顺光逆光打法，服饰道具细节搭配技巧", False),
        ("剪映专业版剪辑节奏卡点技巧：如何剪出让人一秒停留的视频", "声画对齐音效铺底，J-Cut与L-Cut转场在短视频中的实战应用", False),
        ("人像修图调色避坑指南：告别塑料假白磨皮的高级质感肤色", "保留皮肤纹理的高低频磨皮法，色相饱和度精准校正暗沉", False),
        ("大疆Pocket 3夜景实拍测评：一英寸大底低光画质究竟如何？", "云台防抖表现、弱光噪点控制以及D-Log M色彩宽容度测试", True),
        ("胶片摄影入门：从柯达金200到富士C200的胶卷选购与冲扫", "机械单反测光技巧，在不可预知的颗粒感中感受光影沉淀", False),
        ("街头摄影摄影师的心法：如何在不侵犯他人的前提下捕捉决定性瞬间", "预先对焦与盲拍抓拍，在光影几何与市井烟火间寻找冲突美", False),
    ]
}

# Adversarial & Tricky Edge Cases (针对 ModernBERT 语义盲区的经典对抗样本)
ADVERSARIAL_CASES = [
    {
        "title": "牦牛过马路，刹车没刹住 高原牦牛 动物世界精彩瞬间",
        "snippet": "在西藏自驾路上偶遇大群牦牛横穿马路，司机紧急制动有惊无险",
        "ground_truth_track": "萌宠动物",
        "ground_truth_is_clickbait": True,
        "ground_truth_is_ad": False,
        "ground_truth_viral_score": 4,
        "difficulty": "adversarial",
        "notes": "虽然包含西藏自驾，但核心实体与互动为高原动物，极易被纯文本匹配误判为旅游出行"
    },
    {
        "title": "出门旅游尤其国外，千万别开大疆的无感传输",
        "snippet": "无感传输会持续搜索设备消耗电量，且海外公共Wi-Fi下可能引发隐私风险",
        "ground_truth_track": "旅游出行",
        "ground_truth_is_clickbait": True,
        "ground_truth_is_ad": False,
        "ground_truth_viral_score": 4,
        "difficulty": "adversarial",
        "notes": "包含大疆无人机数码产品，但核心语境是旅游出行安全避坑指南"
    },
    {
        "title": "不会白吃的，给你推车省点油钱",
        "snippet": "路遇货车陷车，大家一起帮忙推车脱困，司机师傅硬要塞烟表达感谢",
        "ground_truth_track": "旅游出行",
        "ground_truth_is_clickbait": False,
        "ground_truth_is_ad": False,
        "ground_truth_viral_score": 3,
        "difficulty": "adversarial",
        "notes": "民间互助口语化标题，词面无高频行业关键词"
    },
    {
        "title": "赛里木湖保姆级拍照攻略✨早晚机位",
        "snippet": "天鹅出没位置与日出日落黄金半小时光线，参数建议用长焦70-200mm",
        "ground_truth_track": "摄影剪辑",
        "ground_truth_is_clickbait": False,
        "ground_truth_is_ad": False,
        "ground_truth_viral_score": 3,
        "difficulty": "adversarial",
        "notes": "赛里木湖为著名景区，但主题侧重摄影机位与参数技巧"
    },
    {
        "title": "苹果再一次跳号了！iOS 19 提前曝光",
        "snippet": "爆料显示下一代系统将命名规则调整，跳过版本号直接重构底层架构",
        "ground_truth_track": "科技数码",
        "ground_truth_is_clickbait": True,
        "ground_truth_is_ad": False,
        "ground_truth_viral_score": 5,
        "difficulty": "adversarial",
        "notes": "水果名词多义性，苹果科技数码实体消歧测试"
    },
    {
        "title": "菌子季到了，吃吃喝喝胃根本装不下",
        "snippet": "昆明篆新农贸市场采购各种野生菌，煮一锅鲜掉眉毛的土鸡菌汤火锅",
        "ground_truth_track": "美食探店",
        "ground_truth_is_clickbait": False,
        "ground_truth_is_ad": False,
        "ground_truth_viral_score": 3,
        "difficulty": "adversarial",
        "notes": "涉及云南地理与生物实体，但核心语义为美食探店"
    },
    {
        "title": "谁懂啊！开高速全程90码会被骂吗？老司机聊聊油耗与安全",
        "snippet": "跑长途测试一下省油模式，很多新手在最左侧快车道慢速行驶确实存在隐患",
        "ground_truth_track": "旅游出行",
        "ground_truth_is_clickbait": True,
        "ground_truth_is_ad": False,
        "ground_truth_viral_score": 4,
        "difficulty": "adversarial",
        "notes": "汽车高速行驶与出行驾驶安全"
    },
    {
        "title": "买个这玩意花了1.4w，被家里人骂惨了……",
        "snippet": "为了去川西自驾高海拔徒步买的重装抗风四季帐篷和全套羽绒睡袋",
        "ground_truth_track": "旅游出行",
        "ground_truth_is_clickbait": True,
        "ground_truth_is_ad": False,
        "ground_truth_viral_score": 5,
        "difficulty": "adversarial",
        "notes": "标题极端诱导反差，实为户外装备自驾旅行种草"
    }
]

# Build 1024 balanced samples
all_samples = []
sample_id = 1

# 1. Add adversarial cases
for item in ADVERSARIAL_CASES:
    all_samples.append({
        "id": f"s1_bench_{sample_id:04d}",
        **item
    })
    sample_id += 1

# 2. Add real harvested notes with ground truth classification
for note in existing_notes:
    t = note["title"]
    s = note["snippet"]
    
    # Ground truth labeling heuristics
    t_lower = (t + " " + (s or "")).lower()
    gt_track = "旅游出行"
    if any(w in t_lower for w in ["约拍", "陪拍", "跟拍", "写真", "机位", "无人机", "相机", "哈苏", "索尼", "拍照攻略", "拍摄技巧", "调色"]):
        gt_track = "摄影剪辑"
    elif any(w in t_lower for w in ["gpt", "claude", "deepseek", "mac", "linux", "大模型", "人工智能", "ai", "苹果", "数码", "芯片", "代码", "算法", "键盘", "手机"]):
        gt_track = "科技数码"
    elif any(w in t_lower for w in ["猫", "狗", "宠物", "萌宠", "布偶", "玄猫", "牦牛", "小狗", "小猫", "动物园", "领养"]):
        gt_track = "萌宠动物"
    elif any(w in t_lower for w in ["火锅", "美食", "探店", "餐厅", "咖啡", "甜点", "吃吃喝喝", "菌子", "做饭", "下厨", "小吃"]):
        gt_track = "美食探店"
    elif any(w in t_lower for w in ["护肤", "早c晚a", "粉底", "面膜", "美白", "眼影", "防晒", "面霜", "唇釉"]):
        gt_track = "美妆护肤"
    elif any(w in t_lower for w in ["穿搭", "ootd", "显瘦", "西装", "衬衫", "风衣", "老钱风", "cleanfit"]):
        gt_track = "穿搭时尚"
    elif any(w in t_lower for w in ["跑步", "健身", "减脂", "帕梅拉", "羽毛球", "网球", "普拉提", "滑雪"]):
        gt_track = "运动健康"
    elif any(w in t_lower for w in ["改造", "收纳", "极简", "租房", "卫生间", "toto", "智能家居", "阳台"]):
        gt_track = "家居生活"
    elif any(w in t_lower for w in ["宝宝", "辅食", "产后", "育儿", "母婴", "婴儿", "推车", "绘本"]):
        gt_track = "母婴育儿"
    elif any(w in t_lower for w in ["离职", "面试", "考公", "跳槽", "领导", "职场", "述职", "汇报"]):
        gt_track = "职场成长"
    elif any(w in t_lower for w in ["内耗", "焦虑", "自愈", "独处", "治愈", "心理", "恋爱", "情绪"]):
        gt_track = "情感心理"
    
    is_cb = any(w in t for w in ["千万别", "千万不要", "后悔", "被骂", "踩雷", "坑惨", "竟然", "居然", "谁懂啊", "暴怒", "爆料"])
    is_ad = any(w in t_lower for w in ["推广", "合作", "感谢品牌方", "折扣码", "福利", "优惠券", "自购体验非推广"])
    viral = 2
    if len(t) > 25: viral += 1
    if any(p in t for p in ["？", "?", "！", "!"]): viral += 1
    if is_cb: viral += 1
    viral = min(max(viral, 1), 5)
    
    all_samples.append({
        "id": f"s1_bench_{sample_id:04d}",
        "title": t,
        "snippet": s,
        "ground_truth_track": gt_track,
        "ground_truth_is_clickbait": is_cb,
        "ground_truth_is_ad": is_ad,
        "ground_truth_viral_score": viral,
        "difficulty": "medium",
        "author": note.get("author")
    })
    sample_id += 1

print(f"Total after harvested notes: {len(all_samples)}")

# 3. Augment systematically across all 12 categories until 1024
target_total = 1024
cat_cycle = CATEGORIES.copy()
cat_idx = 0

modifiers = [
    ("深度长文测评", "全面测试了各项核心参数与长期使用寿命"),
    ("避坑排雷实录", "踩过无数坑后的血泪总结，建议收藏"),
    ("保姆级新手全攻略", "从零开始手把手教你少走弯路"),
    ("使用三个月后的真实评价", "优缺点客观分析，拒绝无脑跟风种草"),
    ("私藏小众宝藏推荐", "身边朋友都在问的实用干货"),
    ("治愈人心的生活微小瞬间", "放慢脚步发现日常的美好与松弛感"),
    ("普通人如何高效入门", "没有复杂玄学门槛，只要掌握这几个关键点"),
    ("颠覆认知的实用技巧", "原来看似简单的事情藏着这么多学问")
]

while len(all_samples) < target_total:
    cat = cat_cycle[cat_idx % len(cat_cycle)]
    templates = CATEGORY_CORPUS[cat]
    base_t, base_s, base_cb = random.choice(templates)
    mod_title_prefix, mod_snip_suffix = random.choice(modifiers)
    
    # Generate variations
    rand_type = random.randint(1, 4)
    if rand_type == 1:
        new_title = f"{base_t}｜{mod_title_prefix}"
        new_snip = f"{base_s}。{mod_snip_suffix}"
    elif rand_type == 2:
        new_title = f"谁懂啊！{base_t}，亲测绝了"
        new_snip = base_s
    elif rand_type == 3:
        new_title = f"{base_t}？聊聊我的亲身体验与思考"
        new_snip = f"{base_s}，很多细节只有真正上手才明白"
    else:
        new_title = f"千万别跟风！{base_t}"
        new_snip = f"{base_s}。客观理性评价，不吹不黑"
        
    is_cb = any(w in new_title for w in ["千万别", "千万不要", "谁懂啊", "绝了", "踩雷", "竟然"])
    is_ad = random.random() < 0.08  # ~8% ad ratio
    
    viral = 2
    if len(new_title) > 22: viral += 1
    if any(p in new_title for p in ["？", "?", "！", "!"]): viral += 1
    if is_cb: viral += 1
    viral = min(max(viral, 1), 5)
    
    all_samples.append({
        "id": f"s1_bench_{sample_id:04d}",
        "title": new_title,
        "snippet": new_snip,
        "ground_truth_track": cat,
        "ground_truth_is_clickbait": is_cb,
        "ground_truth_is_ad": is_ad,
        "ground_truth_viral_score": viral,
        "difficulty": "hard" if is_cb else "medium",
        "author": f"User_{random.randint(1000, 9999)}"
    })
    sample_id += 1
    cat_idx += 1

print(f"Final Dataset size: {len(all_samples)} samples.")

# Write main dataset file
out_file = Path("benchmarks/system_one_benchmark/data/benchmark_1024.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=2)

print(f"Saved to {out_file}")

# Distribution summary
from collections import Counter
cat_dist = Counter(s["ground_truth_track"] for s in all_samples)
print("\nCategory Distribution:")
for c, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
    print(f"  {c:<10}: {cnt:4d} ({cnt/len(all_samples):.1%})")

cb_cnt = sum(1 for s in all_samples if s["ground_truth_is_clickbait"])
ad_cnt = sum(1 for s in all_samples if s["ground_truth_is_ad"])
print(f"\nClickbait count: {cb_cnt} ({cb_cnt/len(all_samples):.1%})")
print(f"Commercial Ad count: {ad_cnt} ({ad_cnt/len(all_samples):.1%})")
