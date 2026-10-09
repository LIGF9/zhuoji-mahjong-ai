"""
独山/贵州捉鸡麻将规则引擎（计分按《贵州捉鸡麻将教程》口径）。

计分口径（教程）
----------------
* **唯一计分单位是鸡**：终局所有叫牌（听牌）玩家明牌算鸡，两两互减结算
  （pair[a][b] = c[a] − c[b]）；未叫牌玩家鸡数无效，且要包鸡包杠。
* **鸡的来源**：
  - 手中/副露幺鸡每只 1 分（金鸡时幺鸡 ×2）；
  - 打出的鸡：冲锋鸡 +3（自己第一张打出的鸡牌）、横鸡 +2（该鸡牌种全场第一张非
    冲锋鸡，**以及该种横鸡轮内的跟打**，见下）、幺鸡 +1；
  - **横鸡按牌种独立成轮且可同轮跟打**：某牌种首张非冲锋鸡开轮即横鸡；轮内其他家
    跟打同种鸡也记横鸡；开轮者（或轮内已吃过横鸡者）再次出牌即收口，其后该种鸡
    一律按幺鸡计；
  - 被碰/杠走的冲锋鸡/横鸡：分数归属碰/杠者，但只由打出者单向赔付
    （责任鸡，只在两人之间有效）；
  - 自摸胡牌 +3 鸡；热炮（杠后第一张被胡）+5 鸡；
  - **点炮牌归属**：点炮/热炮的那张牌归胡牌者（从打出者弃牌区移出并入赢家手牌）；
    该张若为鸡牌，其鸡分也一并转给胡牌者（记「捉炮鸡」，打出者不再计）；
  - **一炮多响**：一张牌可同时点炮多名听牌者，各赢家均获得这张牌并各自赔付/结算；
  - **抢杠胡**：补杠（自杠/明杠不可抢）时若该张恰是他人所听，可无视通行证胡牌，
    被抢杠者**鸡分全烧**；**热炮**同理，放热炮者鸡分全烧（自身鸡分作废但仍照付他人）；
  - 听牌玩家每个杠牌 +3 鸡（教程「杠牌计3分」）；
  - 翻鸡：胡牌者翻牌墙顶，点数 +1 同花色为翻鸡牌，听牌者手中/副露每张 +1
    （已打出的不算；上下鸡为可选变体）；
  - 金鸡：翻出九条（翻鸡牌与幺鸡重合）时幺鸡计分 ×2；
  - 牌型奖励（胡牌者，叠加）：清一色 13 / 小七对 13 / 报叫 13 / 单吊 13 /
    龙七对 26 / 大对子 8。
* **包鸡/包杠**：终局未听牌者对自己打出的鸡、自己的杠，向其他每家各赔
  相应分值。**是否为包赔可分别开关**（见 ``DushanConfig.end_baoji`` /
  ``end_baogang``：胡牌局默认包鸡开、包杠关；``huang_baoji`` /
  ``huang_baogang``：流局默认都关）。
* **黄牌（黄庄/荒牌流局）**：**鸡与杠一律不结算**（手中鸡/打出鸡/翻鸡/
  杠分/责任鸡都不计分），只做「未听牌者 → 听牌者」的包赔：每个未听牌者
  向每个听牌者赔 ``自摸分 + 大牌面分``（``huang_baodapai`` 默认开；
  大牌面 = 该听牌者所有听牌张能成的最大牌型分，只有平胡时为 0，此时正好
  等于一个自摸分）。听牌者之间互不结算。``huang_baoji`` / ``huang_baogang``
  （默认关）若打开，未听牌者仍对自己打出的鸡/杠向其余三家赔付。

动作空间与 :class:`zhuoji.rules.ZhuojiGame` 完全一致，
因此编码器与已训练模型可以无缝热启动。
"""
from __future__ import annotations

from dataclasses import dataclass

from .fan import (
    DA_DUI_ZI,
    LONG_QI_DUI,
    PING_HU,
    QING_YI_SE,
    QI_DUI,
    FAN_CN,
    _all_triplets,
    _is_qing_yi_se,
    is_seven_pairs,
    is_tenpai,
    is_winning_hand,
    winning_tiles,
)
from .rules import (
    ANGANG,
    BUGANG,
    HU,
    MELD_KONG_ADDED,
    MELD_KONG_CONCEALED,
    MELD_KONG_EXPOSED,
    MELD_PONG,
    MINGGANG,
    PASS,
    PENG,
    Phase,
    RulesConfig,
    ZhuojiGame,
    meld_is_kong,
)
from .tiles import NUM_TILE_TYPES, YAOJI, rank_of, suit_of, tile_cn

# ---------------------------------------------------------------------------
# 独山分值表（单位：鸡）
# ---------------------------------------------------------------------------
JI_CHONGFENG = 3.0     # 冲锋鸡
JI_HENG = 2.0          # 横鸡
JI_YAO = 1.0           # 幺鸡
KONG_CHICKENS = 3.0    # 听牌者每个杠牌的鸡数
BAO_KONG = 3.0         # 包杠：未听牌者每个杠赔其他每家
ZI_MO_JI = 3.0         # 自摸 +3 鸡
REPAO_JI = 3.0         # 捉热炮 +3 鸡

JI_TAG_BY_VAL = {JI_CHONGFENG: "冲锋鸡", JI_HENG: "横鸡", JI_YAO: "幺鸡"}

DUSHAN_HU_TYPE: dict[str, float] = {
    PING_HU: 0.0,
    DA_DUI_ZI: 8.0,
    QI_DUI: 13.0,
    QING_YI_SE: 13.0,
    "dan_diao": 13.0,     # 单吊
    LONG_QI_DUI: 26.0,
}

DUSHAN_FAN_CN: dict[str, str] = dict(FAN_CN)
DUSHAN_FAN_CN.update({"dan_diao": "单吊"})

CHICKEN_LABELS = ("冲锋鸡", "横鸡", "幺鸡", "手中鸡", "捉炮鸡", "杠牌", "翻鸡",
                  "责任鸡", "包鸡", "包杠", "包大牌面", "自摸", "热炮", "抢杠")

# ---- 翻鸡模式（结算翻鸡 / 开局翻鸡） ----
#: 上鸡：指示牌同花色点数 +1（9 回 1）
FANJI_UP = "up"
#: 下鸡：指示牌同花色点数 -1（1 回 9）
FANJI_DOWN = "down"
#: 上下鸡：+1 与 -1 都算
FANJI_BOTH = "both"
#: 关闭（仅开局翻鸡可用）
FANJI_OFF = "off"

#: 模式 → 中文标签
FANJI_CN = {FANJI_UP: "上鸡", FANJI_DOWN: "下鸡", FANJI_BOTH: "上下鸡", FANJI_OFF: "关闭"}

_FANJI_BOOL = {True: FANJI_BOTH, False: FANJI_OFF}
_FANJI_CN2CODE = {"上鸡": FANJI_UP, "下鸡": FANJI_DOWN, "上下鸡": FANJI_BOTH, "关闭": FANJI_OFF}


def norm_fanji(v: object, *, allow_off: bool) -> str:
    """把各种写法（bool / 中文 / 代码）归一成翻鸡模式字符串。

    ``allow_off=False``（结算翻鸡）时 "off" 与 False 都回落到 "both"。
    """
    if isinstance(v, bool):
        code = _FANJI_BOOL[v]
    elif isinstance(v, str):
        s = v.strip()
        code = _FANJI_CN2CODE.get(s, s.lower())
    elif v is None:
        code = FANJI_OFF
    else:
        code = str(v).lower()
    if code not in (FANJI_UP, FANJI_DOWN, FANJI_BOTH, FANJI_OFF):
        code = FANJI_OFF
    if code == FANJI_OFF and not allow_off:
        code = FANJI_BOTH
    return code


def fanji_tiles_from(flip: int, mode: str) -> set[int]:
    """按模式给出指示牌 ``flip`` 对应的翻鸡牌集合（不含指示牌自身）。"""
    s, r = suit_of(flip), rank_of(flip)
    out: set[int] = set()
    if mode in (FANJI_UP, FANJI_BOTH):
        out.add(s * 9 + (r % 9))                    # 上鸡（9 回 1）
    if mode in (FANJI_DOWN, FANJI_BOTH):
        out.add(s * 9 + ((r - 2) % 9))              # 下鸡（1 回 9）
    return out - {flip}


@dataclass
class DushanConfig(RulesConfig):
    """独山规则档（调参口）。"""

    baojiao_auto: bool = True        # 首张打出后听牌自动报叫
    baojiao_bonus: float = 13.0      # 报叫胡牌 +13 鸡
    # 结算翻鸡（胡牌后翻牌墙顶）："both"=上下鸡（默认）/ "up"=仅上鸡 / "down"=仅下鸡
    jiesuan_fanji: str = "both"
    # 开局翻鸡（开局翻牌墙尾）："off"=关闭（默认）/ "up" / "down" / "both"
    kaiju_fanji: str = "off"
    mantiangji: bool = False         # 满堂鸡：翻鸡牌即使已打出（弃牌区）也计入持有
    jin_ji_multiplier: float = 2.0   # 金鸡倍率（只作用于幺鸡，翻鸡牌本身不翻倍）
    gang_fee: float = 3.0            # 兼容旧字段（新口径不再即时收杠分）
    base_score: float = 1.0
    # ---- 鸡的分值表（教程 10.1，网页设置可改；单位：鸡） ----
    ji_hold: float = 1.0             # 手中/副露鸡：每张
    ji_discard_plain: float = 1.0    # 打出的普通鸡：每张（2026-10-08 口径修正：原为 0）
    ji_chongfeng: float = 3.0        # 冲锋鸡（自己打出的第一张鸡牌，打出即计）
    ji_heng: float = 2.0             # 横鸡（该鸡牌种全场第一张被打出，打出即计）
    ji_fanji: float = 1.0            # 翻鸡牌：每张（听牌者手中/副露）
    # ---- 可调分值表（网页设置可改；单位：鸡） ----
    kong_chickens: float = 3.0       # 每个杠牌的杠分（明杠由点杠者付，暗杠/补杠全场付）
    zi_mo: float = 3.0               # 自摸（全场每家付）
    dian_pao: float = 3.0            # 点炮（点炮者独付）
    gang_kai: float = 5.0            # 杠上开花（全场每家付）
    gang_pao: float = 5.0            # 热炮（杠后第一张被胡，放热炮者独付且鸡分全烧）
    da_dui_zi: float = 5.0           # 大对子
    qing_yi_se: float = 10.0         # 清一色
    qi_dui: float = 10.0             # 小七对
    long_qi_dui: float = 20.0        # 龙七对
    dan_diao: float = 10.0           # 独钓（单吊）
    # ---- 未听牌包赔开关（网页设置可改，全局默认） ----
    end_baoji: bool = True           # 有人胡牌结束：未听牌者包鸡（默认开）
    end_baogang: bool = False        # 有人胡牌结束：未听牌者包杠（默认关）
    huang_baoji: bool = False        # 流局（黄庄）：未听牌者包鸡（默认关）
    huang_baogang: bool = False      # 流局（黄庄）：未听牌者包杠（默认关）
    huang_baodapai: bool = True      # 流局：未听牌者包听牌者的非平胡牌面（默认开）

    def __post_init__(self) -> None:
        # 兼容旧写法：布尔 / 中文标签 / 代码字符串，统一归一为翻鸡模式
        self.jiesuan_fanji = norm_fanji(self.jiesuan_fanji, allow_off=False)
        self.kaiju_fanji = norm_fanji(self.kaiju_fanji, allow_off=True)


class DushanGame(ZhuojiGame):
    """一局独山麻将。动作空间与 ZhuojiGame 完全一致。"""

    def __init__(self, config: DushanConfig | None = None, seed: int | None = None,
                 dealer: int | None = None, dealer_streak: int = 0):
        super().__init__(config or DushanConfig(), seed=seed,
                         dealer=dealer, dealer_streak=dealer_streak)

    # -- 独山状态 -----------------------------------------------------------
    def _init_dushan_state(self) -> None:
        self.baojiao = [False] * 4              # 报叫标记
        self.first_discard_done = [False] * 4   # 是否已打过第一张牌
        self.hengji_opened = False              # 全场第一张鸡（横鸡）是否已出现
        self.hengji_starter: int | None = None  # 本局第一个开出横鸡轮的玩家
        self.ji_events: list[tuple[int, float, str]] = []   # (打出者, 基础分, 类型名)
        # 横鸡按鸡牌种独立成轮（2026-10-08 用户口径：幺鸡与开局翻鸡种各算各的，
        # 互不影响）：_hengji_done[牌种] = 该种鸡的横鸡已出现（本局不再开新轮）。
        self._hengji_done: set[int] = set()
        # 正在开放中的「横鸡轮」（2026-10-09 用户口径：横鸡可以同轮跟打）：
        #   _hengji_rounds[牌种] = {"start": 开轮者, "seats": 轮内已吃横鸡的玩家集合}
        # 轮内任何玩家跟打同种鸡都算横鸡；开轮者或轮内已打过者再次出牌 → 该轮收口。
        self._hengji_rounds: dict[int, dict] = {}
        self._ji_tag_of_last = ["幺鸡"] * 4     # 各家最近一次打出的鸡牌类型名
        # ((玩家, 牌), 被碰/杠鸡的类型名) —— 存类型名而非分值，分值改由配置换算
        self._ji_val_by_meld: dict[tuple[int, int], str] = {}
        self.claimed_ji: list[tuple[int, int, str, bool]] = []   # (打出者, 碰/杠者, 类型名, 是否明杠)
        # 捉炮鸡：点炮/热炮的那张牌若是鸡牌，其鸡分归胡牌者（打出者不再计）。
        # (胡牌者, 分值, 类型名, 牌)
        self.captured_ji: list[tuple[int, float, str, int]] = []
        # 一炮多响：本次响应中已表态胡牌的玩家（等所有可胡者表态完毕再统一结算）
        self._respond_hu_votes: list[int] = []
        self._n_hu: int = 0
        self.repao_discard = False              # 当前弃牌是否热炮（杠后第一张）
        self._after_gang_draw = False           # 本次进牌来自杠后摸牌（杠上开花）
        self._any_discard = False               # 天胡判定用
        self.ji_tiles: set[int] = {YAOJI}       # 本局的鸡牌种（幺鸡 + 开局翻鸡新增）
        self.kaiju_flip: int | None = None      # 开局翻出的鸡牌指示牌（None=未启用）

    # -- 横鸡轮（只读派生） --------------------------------------------------
    @property
    def hengji_active(self) -> bool:
        """是否有任一种鸡牌正处于「横鸡轮」内（该种鸡此时跟打也算横鸡）。"""
        return bool(self._hengji_rounds)

    @property
    def hengji_species(self) -> list[int]:
        """当前仍开放横鸡轮的鸡牌种（升序）。"""
        return sorted(self._hengji_rounds)

    def ji_name_if_discarded(self, p: int, tile: int) -> str:
        """玩家 ``p`` 现在打出 ``tile`` 会记成什么鸡牌：冲锋鸡/横鸡/幺鸡。

        与 :meth:`_step_discard` 的判定同源，供 UI/教练提示复用（不改动状态）。
        """
        if tile not in self.ji_tiles:
            return ""
        if not self.first_discard_done[p]:
            return "冲锋鸡"
        if tile in self._hengji_done:
            rnd = self._hengji_rounds.get(tile)
            return "横鸡" if (rnd is not None and p not in rnd["seats"]) else "幺鸡"
        return "横鸡"

    def _close_hengji_rounds(self, p: int) -> None:
        """弃牌前收口：开轮者或轮内已吃横鸡者再次出牌 → 该种横鸡轮结束。

        与本次打出的牌无关（旧版实现口径），所以「收口那一手」打出的鸡牌按普通鸡计。
        """
        for sp in [s for s, r in self._hengji_rounds.items() if p in r["seats"]]:
            del self._hengji_rounds[sp]

    def _start(self, dealer: int | None) -> None:  # noqa: D102
        # 独山状态必须先于基类初始化：_enter_self_phase 会用到报叫锁定
        self._init_dushan_state()
        super()._start(dealer)
        # 开局翻鸡：从牌墙尾部翻一张指示牌。**只翻开查看，不移出牌墙**
        # （2026-10-08 用户口径修正：这张牌本局仍留在牌墙里，仍可能被摸到）。
        # 按 kaiju_fanji 模式（上鸡/下鸡/上下鸡）成为本局新增鸡牌种。
        if self.cfg.kaiju_fanji != FANJI_OFF and self.wall:
            flip = self.wall[-1]
            self.kaiju_flip = flip
            self.ji_tiles.update(fanji_tiles_from(flip, self.cfg.kaiju_fanji))

    # -- 鸡分值换算（全部走配置，网页设置可调） ------------------------------
    def ji_value(self, tag: str) -> float:
        """按当前配置把「打出的鸡」类型名换算成分值。

        - 冲锋鸡 → ``cfg.ji_chongfeng``（默认 3）
        - 横鸡   → ``cfg.ji_heng``（默认 2）
        - 幺鸡/普通鸡 → ``cfg.ji_discard_plain``（默认 1，2026-10-08 修正：原为 0）
        """
        cfg: DushanConfig = self.cfg  # type: ignore[assignment]
        if tag == "冲锋鸡":
            return float(cfg.ji_chongfeng)
        if tag == "横鸡":
            return float(cfg.ji_heng)
        return float(cfg.ji_discard_plain)

    # -- 报叫锁定：不可碰杠 ---------------------------------------------------
    def _can_peng(self, p: int, tile: int) -> bool:
        if self.baojiao[p]:
            return False
        return super()._can_peng(p, tile)

    def _can_minggang(self, p: int, tile: int) -> bool:
        if self.baojiao[p]:
            return False
        return super()._can_minggang(p, tile)

    def _self_kong_options(self, p: int) -> list[tuple[str, int]]:
        if self.baojiao[p]:
            return []
        return super()._self_kong_options(p)

    # -- 牌型判定（独山口径，叠加制） -----------------------------------------
    def _win_type_set(self, hand14: list[int], melds,
                      win_tile: int | None = None) -> set[str]:
        """对 14 张成胡手牌判定全部可叠加牌型。

        ``win_tile``：这张牌的牌型要看「是不是靠它凑成四张同牌」——
        龙七对必须是**最后听/胡的那张牌**补上第四张；四张早就攥在手里、
        胡的是别的对子时只能算小七对（2026-10-08 用户口径）。
        ``win_tile=None``（听牌分析等场合只有牌面）时保留宽松口径。
        """
        types: set[str] = set()
        n_melds = len(melds)
        seven = n_melds == 0 and is_seven_pairs(hand14)
        normal = is_winning_hand(hand14, melds)
        if not (seven or normal):
            return types
        if seven:
            if win_tile is None:
                dragon = any(x == 4 for x in hand14)
            else:
                dragon = int(hand14[int(win_tile)]) == 4
            types.add(LONG_QI_DUI if dragon else QI_DUI)
        else:
            all_triplet_melds = all(m[0] != 0 for m in melds)   # 无吃（捉鸡/独山本无吃）
            if n_melds == 4 and all_triplet_melds and sum(hand14) == 2:
                types.add("dan_diao")          # 单钓将与大对子互斥（参照工程口径）
            elif _all_triplets(hand14, melds):
                types.add(DA_DUI_ZI)
        if _is_qing_yi_se(hand14, melds):
            types.add(QING_YI_SE)
        if not types:
            types.add(PING_HU)
        return types

    def _has_passport(self, p: int, hand14: list[int] | None = None,
                      win_tile: int | None = None) -> bool:
        """独山通行证：杠 / 报叫 / 胡牌张成非平胡牌型。

        ``hand14``（已并入胡牌张的 14 张暗手）非 None 时按**该胡牌张**判定：
        这张牌成的是非平胡（大对子/七对/清一色/单吊/龙七对）→ 无需杠/报叫
        也可点胡；成的是平胡 → 必须有杠或报叫（2026-10-08 用户口径修正：
        旧实现「任一等待张成非平胡即全手有通行证」过宽，导致听
        4万平胡/5万大对子 的混合听牌无豆也能平胡点炮）。
        ``hand14`` 为 None 时保留旧口径（按是否存在非平胡听张），供兼容调用。
        """
        if any(meld_is_kong(mt) for mt, _, _ in self.melds[p]):
            return True
        if self.baojiao[p]:
            return True
        if hand14 is not None:
            return bool(self._win_type_set(hand14, self.melds[p], win_tile) - {PING_HU})
        for t in winning_tiles(self.hands[p], self.melds[p]):
            h = list(self.hands[p])
            h[t] += 1
            if self._win_type_set(h, self.melds[p], t) - {PING_HU}:
                return True
        return False

    def can_ron(self, p: int, tile: int) -> bool:  # noqa: D102
        if self.cfg.enable_pass_lock and tile in self.pass_lock[p] \
                and not self.repao_discard:
            return False
        h = list(self.hands[p])
        h[tile] += 1
        if not is_winning_hand(h, self.melds[p]):
            return False
        # 抢杠胡：补杠被抢时无视通行证（听该张即可胡）
        rob = getattr(self, "_rob_kong", None)
        if rob is not None and int(rob[1]) == int(tile):
            return True
        # 通行证按「这张胡牌张成的牌型」逐张判定
        if self._has_passport(p, h, tile):
            return True
        # 热炮：任何人都可无通行证吃胡
        if self.repao_discard:
            return True
        # 报叫玩家的弃牌：无需通行证
        if self.last_discard and self.baojiao[self.last_discard[0]]:
            return True
        return False

    def _can_rob_kong(self, p: int, tile: int) -> bool:
        """抢杠胡资格：补杠的这张牌恰好是自己在听的牌即可胡（无视通行证）。"""
        if self.cfg.enable_pass_lock and tile in self.pass_lock[p]:
            return False
        h = list(self.hands[p])
        h[tile] += 1
        return is_winning_hand(h, self.melds[p])

    # -- 抢杠：补杠可被抢（无视通行证），自杠/明杠不可抢 ------------------------
    def _step_self_kong(self, action) -> None:  # noqa: D102
        if self._rob_kong is None and action.kind == BUGANG:
            p = self.current
            t = action.tile
            if self.cfg.enable_qiang_gang:
                robbers = [q for q in self._opponents(p) if self._can_rob_kong(q, t)]
                if robbers:
                    self._rob_kong = (robbers[0], t)
                    self._rob_kong_konger = p
                    self.phase = Phase.SELF_KONG
                    return
            self._do_bugang(p, t)
            return
        super()._step_self_kong(action)

    # -- 出牌阶段：冲锋鸡 / 横鸡轮 / 热炮 / 自动报叫 ---------------------------
    def _step_discard(self, action) -> None:  # noqa: D102
        p = self.current
        if action.kind == HU:
            # 防线：碰牌后未摸牌（drawn=False）不可自摸胡——历史上编码器掩码曾漏判，
            # 导致模型在该局面推荐/采样出非法自摸胡，静默产生假胡污染训练数据
            if not (self.drawn and self.can_tsumo(p)):
                raise ValueError(
                    f"非法自摸胡（玩家 {p}）：drawn={self.drawn}，"
                    f"can_tsumo={self.can_tsumo(p)}（碰后未摸牌须先出牌）")
            self._settle_win(p, -1, is_tsumo=True)
            return
        t = action.tile
        # 热炮 = 杠后第一张弃牌（复用基类 gang_discard 标记）
        self.repao_discard = self.gang_discard[p]
        self.hands[p][t] -= 1
        self.discards[p].append(t)
        self._any_discard = True

        # 横鸡轮收口：开轮者 / 轮内已吃横鸡者再次出牌 → 该种轮结束（与本次牌无关）
        self._close_hengji_rounds(p)

        # 鸡牌判定（教程口径 + 2026-10-08 分种轮 + 2026-10-09 同轮跟打）：
        #   冲锋鸡 = 自己打出的第一张牌，若为鸡牌（按其鸡牌种记账，每家最多一次）；
        #   横鸡   = 该鸡牌种全场第一张非冲锋鸡，**以及该种「横鸡轮」内其他家跟打的同种鸡牌**
        #            （每种鸡独立成轮：幺鸡的轮不影响开局翻鸡种；一轮收口后该种再打即普通鸡）；
        #   其余为幺鸡（普通鸡）。
        if t in self.ji_tiles:
            if not self.first_discard_done[p]:
                tag = "冲锋鸡"
            elif t in self._hengji_done:
                # 该种横鸡已出现过：轮内跟打仍算横鸡，轮已收口则按普通鸡
                tag = "横鸡" if t in self._hengji_rounds else "幺鸡"
            else:
                self._hengji_done.add(t)
                self._hengji_rounds[t] = {"start": p, "seats": {p}}
                if not self.hengji_opened:
                    self.hengji_opened = True
                    self.hengji_starter = p
                tag = "横鸡"
            # 轮内参与者登记（含轮内打出的冲锋鸡）：该家再次出牌即触发收口
            rnd = self._hengji_rounds.get(t)
            if rnd is not None:
                rnd["seats"].add(p)
            val = self.ji_value(tag)
            self.ji_events.append((p, val, tag))
            self._ji_tag_of_last[p] = tag

        # 自动报叫：第一张打出后已听牌
        if not self.first_discard_done[p]:
            self.first_discard_done[p] = True
            if self.cfg.baojiao_auto and is_tenpai(self.hands[p], self.melds[p]):
                self.baojiao[p] = True
                self.log.append((p, "报叫"))

        self._update_tenpai(p)
        self.last_discard_was_gang = self.gang_discard[p]
        self.gang_discard[p] = False
        self.last_discard = (p, t)
        self._begin_respond(p, t)

    def _gang_draw(self, p: int) -> None:  # noqa: D102
        super()._gang_draw(p)
        self._after_gang_draw = True

    def _next_player(self, from_p: int) -> None:  # noqa: D102
        self.repao_discard = False
        self._after_gang_draw = False   # 普通流转摸牌不再是杠后摸牌
        super()._next_player(from_p)

    # -- 响应阶段：一炮多响收集 / 记录被碰鸡牌类型（责任分用） ------------------
    def _begin_respond(self, discarder: int, tile: int) -> None:  # noqa: D102
        """开始响应：所有可胡者依次表态（支持一炮多响），其后才轮到碰/杠者。"""
        others = self._opponents(discarder)
        hu_list = [q for q in others if self.can_ron(q, tile)]
        claim_list = [q for q in others if q not in hu_list
                      and (self._can_peng(q, tile) or self._can_minggang(q, tile))]
        order = hu_list + claim_list
        self._respond_hu_votes = []
        self._n_hu = len(hu_list)
        if order:
            self._respond_order = order
            self._respond_idx = 0
            self.phase = Phase.RESPOND
        else:
            self._next_player(discarder)

    def _advance_respond(self, discarder: int, tile: int) -> None:
        """推进到下一个响应者；可胡者全部表态完毕且有人胡 → 一炮多响统一结算。"""
        self._respond_idx += 1
        if self._respond_hu_votes and self._respond_idx >= self._n_hu:
            winners = list(self._respond_hu_votes)
            self._respond_order = []
            self._respond_idx = 0
            self._respond_hu_votes = []
            self._n_hu = 0
            self._settle_win_multi(winners, tile, is_tsumo=False, loser=discarder)
            return
        if self._respond_idx >= len(self._respond_order):
            self._respond_order = []
            self._respond_idx = 0
            self._respond_hu_votes = []
            self._n_hu = 0
            self._next_player(discarder)

    def _step_respond(self, action) -> None:  # noqa: D102
        p = self._current_responder()
        discarder, tile = self.last_discard if self.last_discard else (None, -1)
        if action.kind == HU:
            # 一炮多响：先记票，待所有可胡者表态完毕后统一结算
            if p not in self._respond_hu_votes:
                self._respond_hu_votes.append(p)
            self._advance_respond(discarder, tile)
            return
        if action.kind == PASS:
            self.pass_lock[p].add(tile)
            self._advance_respond(discarder, tile)
            return
        claim = action.kind in (PENG, MINGGANG)
        ji_tag = self._ji_tag_of_last[discarder] if discarder is not None else "幺鸡"
        super()._step_respond(action)
        self._respond_hu_votes = []
        self._n_hu = 0
        if claim:
            if tile in self.ji_tiles:
                # 记录被碰/杠的鸡（含普通鸡）：该张从打出者名下**转移**给碰/杠者，
                # 打出者另付溢价（= 该张打出值 − 手中鸡单值，见 _claimed_ji_rows）
                self._ji_val_by_meld[(p, tile)] = ji_tag
                self.claimed_ji.append((discarder, p, ji_tag, action.kind == MINGGANG))
            self.repao_discard = False

    # -- 翻鸡（胡牌者翻牌墙顶；按结算翻鸡模式取 +1 / -1 同花色牌） --------------
    def _dushan_chicken_state(self) -> tuple[bool, set[int]]:
        """返回 (是否金鸡, 翻鸡牌集合)。

        模式见 ``cfg.jiesuan_fanji``：up=仅上鸡(+1)、down=仅下鸡(-1)、both=上下鸡。
        金鸡：翻出的牌按当前模式算出的翻鸡牌里有幺鸡
        （上鸡翻九条 9→1；下鸡翻二条 2→1），此时幺鸡计分翻倍。
        """
        if self.wall_left <= 0:
            return False, set()
        mode = self.cfg.jiesuan_fanji
        flip = self.wall[self.wall_pos]
        fanji = fanji_tiles_from(flip, mode)
        gold = YAOJI in fanji
        return gold, fanji

    # -- 结算（教程口径） ------------------------------------------------------
    def _own_ji_after_claims(self) -> tuple[list[list[float]], list[list[str]]]:
        """各家「自己打出的鸡」（值/标签）。

        被碰/杠走的鸡（含普通鸡）从打出者名单中抵消——那张牌已经**转移**给
        碰/杠者，由碰/杠者的副露张数在「手中鸡」里计（碰 3 / 杠 4 张）。
        """
        vals: list[list[float]] = [[] for _ in range(4)]
        tags: list[list[str]] = [[] for _ in range(4)]
        for pp, val, tag in self.ji_events:
            vals[pp].append(val)
            tags[pp].append(tag)
        for ev in self.claimed_ji:
            discarder, tag = int(ev[0]), str(ev[2])
            for i in range(len(tags[discarder]) - 1, -1, -1):
                if tags[discarder][i] == tag:
                    vals[discarder].pop(i)
                    tags[discarder].pop(i)
                    break
        return vals, tags

    def _chicken_counts(self, tenpai: list[bool], fanji: set[int], jin: float,
                        void: int | None = None):
        """统计各家鸡数（仅听牌者计鸡）。

        返回 (c, items, own_vals, own_tags)；items[p] = [(label, info, val), ...]
        为该玩家鸡数的来源明细（冲锋鸡/横鸡/手中鸡/捉炮鸡/翻鸡…）。

        ``void``：鸡分全烧的玩家（热炮放炮者 / 被抢杠者）——其自身鸡分一律作废
        （手中鸡/打出鸡/捉炮鸡/翻鸡全不计），但仍照付其他玩家的鸡分。

        分值口径（2026-10-08 用户确认，全部可在网页设置里改）：
        手中/副露鸡 ``cfg.ji_hold``、打出的普通鸡 ``cfg.ji_discard_plain``、
        冲锋鸡 ``cfg.ji_chongfeng``、横鸡 ``cfg.ji_heng``、翻鸡 ``cfg.ji_fanji``；
        金鸡时**只有幺鸡**乘 ``cfg.jin_ji_multiplier``。
        """
        cfg: DushanConfig = self.cfg  # type: ignore[assignment]
        c = [0.0] * 4
        items: list[list[tuple[str, str, float]]] = [[] for _ in range(4)]

        def add(p: int, val: float, label: str, info: str = "") -> None:
            if val <= 0:
                return
            c[p] += val
            items[p].append((label, info, val))

        own_vals, own_tags = self._own_ji_after_claims()
        # 打出的鸡**全部计分**（2026-10-08 口径修正）：冲锋鸡 / 横鸡 / 普通鸡
        # （打出的普通鸡 = cfg.ji_discard_plain，默认 1，原来按 0 处理）。
        # 被碰/杠走的那张已由 _own_ji_after_claims 从打出者名下转移走，
        # 不会与碰/杠者副露里的同张重复计算。

        # 捉炮鸡：点炮张若是鸡牌，该张已并入捉炮者手牌，此处按张全额计分
        # （手中鸡已把这张算作 1×ji_hold，故先扣张数避免重复，再按原值全额计入）
        cap_cnt = [0] * 4
        cap_yao = [0] * 4
        for (w, _val, _tag, _tile) in self.captured_ji:
            cap_cnt[w] += 1
            if int(_tile) == YAOJI:
                cap_yao[w] += 1

        for p in range(4):
            if not tenpai[p]:
                continue
            # 1) 打出的鸡：冲锋鸡 / 横鸡 / 普通鸡（含被碰/杠走的已扣除）
            for val, tag in zip(own_vals[p], own_tags[p]):
                add(p, val, tag, "打出")
            # 2) 手中/副露鸡牌（幺鸡 + 开局翻鸡新增种）；金鸡时**只有幺鸡** ×jin。
            #    被碰/杠走的鸡牌牌面同样计入（碰 3 / 杠 4 张，对所有人有效）；
            #    打出者的责任溢价单独结算（见 _claimed_ji_rows）
            hold = float(cfg.ji_hold)
            n_yao = self.hands[p][YAOJI] + self.melded_count(p, YAOJI) - cap_yao[p]
            n_oth = sum(self.hands[p][t] + self.melded_count(p, t)
                        for t in self.ji_tiles if t != YAOJI)
            n_yao = max(0, n_yao)
            n_hand = n_yao + max(0, n_oth)
            n_val = (n_yao * jin + max(0, n_oth)) * hold
            if n_val > 0:
                add(p, n_val, "手中鸡",
                    f"{n_hand:g} 只" + ("（金鸡×2）" if jin > 1 and n_yao > 0 else ""))
            # 3) 杠分：不再计入鸡数池（明杠只对点杠者、暗杠/补杠对全场，见 _settle_*）
            # 4) 翻鸡牌（不乘金鸡）；满堂鸡：弃牌区里的翻鸡牌同样计入
            if fanji:
                nf = sum(self.hands[p][ft] + self.melded_count(p, ft) for ft in fanji)
                extra_info = ""
                if self.cfg.mantiangji:
                    nd = sum(self.discards[p].count(ft) for ft in fanji)
                    if nd:
                        nf += nd
                        extra_info = f"（含弃牌 {nd:g} 张）"
                if nf:
                    # 翻鸡牌种由前端渲染成牌图（res.fanji_tiles），这里只给数量
                    add(p, float(nf) * float(cfg.ji_fanji), "翻鸡",
                        f"共持有 {nf:g} 张" + extra_info)
        # 5) 捉炮鸡：点炮/热炮张归胡牌者，按其鸡牌类型全额计入（冲锋3/横2/普通1）
        for (w, val, tag, tile) in self.captured_ji:
            add(w, float(val), "捉炮鸡", f"{tag}·{tile_cn(tile)}")
        # 全烧：该玩家自身鸡分一律作废（仍照付他人）
        if void is not None and 0 <= int(void) < 4:
            v0 = int(void)
            c[v0] = 0.0
            items[v0] = []
        return c, items, own_vals, own_tags

    # -- 牌型/听牌辅助 ---------------------------------------------------------
    def _hu_type_value(self, tp: str) -> float:
        """牌型奖励鸡数（读设置分值表）。"""
        cfg: DushanConfig = self.cfg  # type: ignore[assignment]
        return {
            PING_HU: 0.0,
            DA_DUI_ZI: cfg.da_dui_zi,
            QI_DUI: cfg.qi_dui,
            QING_YI_SE: cfg.qing_yi_se,
            "dan_diao": cfg.dan_diao,
            LONG_QI_DUI: cfg.long_qi_dui,
        }.get(tp, 0.0)

    def _tenpai_flag(self, p: int) -> bool:
        """听牌判定（教学口径：打张后能胡即算听）。

        标准牌数（暗手 = (4-副露数)*3+1）直接用 winning_tiles；
        非标准牌数（杠后未及补牌 / 被抢杠 / 终局持 14 张等）按
        「打出任意一张后是否听牌」容错判定，避免把明显听牌的手判成未听。
        """
        if is_tenpai(self.hands[p], self.melds[p]):
            return True
        n = len(self.melds[p])
        if sum(self.hands[p]) == (4 - n) * 3 + 1:
            return False
        h = [int(x) for x in self.hands[p]]
        for d in range(NUM_TILE_TYPES):
            if h[d] <= 0:
                continue
            h[d] -= 1
            ok = bool(winning_tiles(h, self.melds[p]))
            h[d] += 1
            if ok:
                return True
        return False

    def ting_info(self, p: int, drop: int | None = None) -> tuple[list[int], list[str]]:
        """听牌信息：([可胡的牌], [牌型名集合])。未听牌返回空。

        ``drop``：先把这张牌从暗手去掉再算（赢家终局亮牌用——结算时胡牌张
        已并入赢家手牌形成 14 张，须去掉后才能还原真实等待张，否则会走到
        「打张后兜底」分支给出误导性的假听张）。

        暗手为标准牌数时直接判；非标准牌数（杠后未补牌、被抢杠等）按打张后
        口径：打出某张后听牌，则该听的牌即等待张。
        """
        base = [int(x) for x in self.hands[p]]
        if drop is not None and 0 <= int(drop) < NUM_TILE_TYPES and base[int(drop)] > 0:
            base[int(drop)] -= 1
        melds = self.melds[p]
        tiles = winning_tiles(base, melds)
        if not tiles and sum(base) != (4 - len(melds)) * 3 + 1:
            for d in range(NUM_TILE_TYPES):
                if base[d] <= 0:
                    continue
                base[d] -= 1
                wt = winning_tiles(base, melds)
                if wt:
                    tiles = wt
                    break
                base[d] += 1
            else:
                return [], []
        types: set[str] = set()
        for t in tiles:
            h = list(base)
            h[t] += 1
            types |= self._win_type_set(h, melds, t)
        named = [DUSHAN_FAN_CN.get(tp, tp) for tp in sorted(types) if tp != PING_HU]
        if not named and tiles:
            named = ["平胡"]
        return sorted(tiles), named

    def _max_hu_type(self, p: int) -> tuple[float, str]:
        """听牌者「可达最大牌面」：(分值, 牌型名)。

        遍历该玩家所有可胡张，取其中最高的牌型分（平胡为 0）。流局包大牌面用。
        """
        base = [int(x) for x in self.hands[p]]
        melds = self.melds[p]
        tiles = winning_tiles(base, melds)
        if not tiles and sum(base) != (4 - len(melds)) * 3 + 1:
            for d in range(NUM_TILE_TYPES):
                if base[d] <= 0:
                    continue
                base[d] -= 1
                wt = winning_tiles(base, melds)
                if wt:
                    tiles = wt
                    break
                base[d] += 1
        best_v, best_cn = 0.0, ""
        for t in tiles:
            h = list(base)
            h[t] += 1
            types = self._win_type_set(h, melds, t)
            v = float(sum(self._hu_type_value(tp) for tp in types))
            if v > best_v:
                best_v = v
                best_cn = "+".join(
                    DUSHAN_FAN_CN.get(tp, tp) for tp in sorted(types) if tp != PING_HU)
        return best_v, best_cn

    def _claimed_ji_rows(self) -> list[tuple[int, int, float, str]]:
        """碰/杠鸡牌的溢价结算行 [(碰/杠者, 打出者, 溢价, 鸡类名)]。

        口径（2026-10-08 用户确认）：B 碰/杠 A 打出的鸡——那张牌**转移**给 B，
        按副露张数计入 B 的手中鸡（碰 3 / 杠 4，对所有人有效）；A 另付一笔溢价，
        **溢价 = 该张打出的分值 − 手中鸡单值**：
        冲锋鸡 3−1=2 / 横鸡 2−1=1 / 普通鸡 1−1=0（普通鸡只转移，无溢价）。

        AB 之间合计：碰冲锋 5（面 3+溢 2）/ 碰横 4（面 3+溢 1）/
        杠冲锋 9（=杠分 3+面 4+溢 2）/ 杠横 8（=杠分 3+面 4+溢 1）。
        """
        cfg: DushanConfig = self.cfg  # type: ignore[assignment]
        rows: list[tuple[int, int, float, str]] = []
        for ev in self.claimed_ji:
            discarder, claimer, tag = int(ev[0]), int(ev[1]), str(ev[2])
            premium = self.ji_value(tag) - float(cfg.ji_hold)
            if premium > 0:
                rows.append((claimer, discarder, premium, tag))
        return rows

    def _transfer_discarded_ji(self, discarder: int, winners: list[int], tile: int) -> None:
        """点炮/热炮的那张牌若是鸡牌：把该张鸡分从打出者转给各胡牌者。

        该张是打出者最近一次弃牌（点炮张），故其鸡事件必为该玩家最后一条。
        转移后打出者不再计这张的鸡分，胡牌者按牌型全额计「捉炮鸡」。
        """
        if tile not in self.ji_tiles:
            return
        idx = -1
        for i in range(len(self.ji_events) - 1, -1, -1):
            if int(self.ji_events[i][0]) == int(discarder):
                idx = i
                break
        if idx < 0:
            return
        val, tag = float(self.ji_events[idx][1]), str(self.ji_events[idx][2])
        self.ji_events.pop(idx)
        for w in winners:
            self.captured_ji.append((int(w), val, tag, int(tile)))

    def _settle_win(self, winner: int, win_tile: int, is_tsumo: bool,
                    loser: int | None = None, rob_kong: bool = False) -> None:
        self._settle_win_multi([int(winner)], win_tile, is_tsumo, loser, rob_kong)

    def _settle_win_multi(self, winners, win_tile, is_tsumo: bool,
                          loser: int | None = None, rob_kong: bool = False) -> None:
        """胡牌结算（支持一炮多响：``winners`` 可含多名赢家）。"""
        cfg: DushanConfig = self.cfg  # type: ignore[assignment]
        winners = [int(w) for w in winners]
        primary = winners[0]

        # ---- 胡牌张归属：点炮/热炮从打出者弃牌区移出（捉炮者的牌）；
        #      抢杠从被抢者手中移出；自摸本就在手牌 ----
        wt_i = int(win_tile) if isinstance(win_tile, (int, float)) and int(win_tile) >= 0 else -1
        if wt_i >= 0:
            for w in winners:
                self.hands[w][wt_i] += 1
            if not is_tsumo and loser is not None:
                if rob_kong:
                    if self.hands[loser][wt_i] > 0:
                        self.hands[loser][wt_i] -= 1
                else:
                    if self.discards[loser] and self.discards[loser][-1] == wt_i:
                        self.discards[loser].pop()
                    self._transfer_discarded_ji(loser, winners, wt_i)

        types_by = {w: self._win_type_set(self.hands[w], self.melds[w],
                                          wt_i if wt_i >= 0 else None) for w in winners}

        gold, fanji = self._dushan_chicken_state()
        jin = cfg.jin_ji_multiplier if gold else 1.0
        detail: list[tuple] = []
        fanji_flip = self.wall[self.wall_pos] if self.wall_left > 0 else None
        if fanji:
            detail.append((-1, "翻鸡", tile_cn(fanji_flip) + ("（金鸡）" if gold else ""), 0.0))

        pair = [[0.0] * 4 for _ in range(4)]
        pair_det: dict = {}

        def pflow(a: int, b: int, val: float, label: str, info: str = "",
                  who: int | None = None) -> None:
            """a 收、b 付。``who`` = 该分项归属（责任方口径，默认记在收方 a）。"""
            if val <= 0:
                return
            pair[a][b] += val
            pair_det.setdefault((a, b), []).append(
                (label, info, val, a if who is None else int(who)))

        # 终局听牌（胡牌者视为叫牌）
        tenpai = [self._tenpai_flag(p) for p in range(4)]
        for w in winners:
            tenpai[w] = True

        # ---- 全烧：热炮放炮者 / 被抢杠者（自身鸡分作废，但仍照付他人） ----
        void_player: int | None = None
        if not is_tsumo and loser is not None and (rob_kong or self.repao_discard):
            void_player = int(loser)

        # ---- 鸡数统计（手中/打出/捉炮/翻鸡，对所有人有效；两两互减） ----
        c, items, own_vals, own_tags = self._chicken_counts(
            tenpai, fanji, jin, void=void_player)
        for p in range(4):
            for label, info, val in items[p]:
                detail.append((p, label, info, val))
        if void_player is not None:
            detail.append((void_player, "全烧", "鸡分作废", 0.0))

        # ---- 胡牌方式：自摸/杠上开花全场每家付；点炮/热炮/抢杠由放炮者独付 ----
        if is_tsumo:
            how, how_val = (("杠上开花", cfg.gang_kai) if self._after_gang_draw
                            else ("自摸", cfg.zi_mo))
        elif rob_kong:
            how, how_val = "抢杠", cfg.dian_pao
        elif self.repao_discard:
            how, how_val = "热炮", cfg.gang_pao
        else:
            how, how_val = "点炮", cfg.dian_pao
        if is_tsumo:
            payers = [q for q in range(4) if q != primary]
        else:
            payers = [int(loser)] if loser is not None else []
        how_tag = how + (f"（{len(winners)}家）" if len(winners) > 1 else "")
        for w in winners:
            for q in payers:
                if q != w:
                    pflow(w, q, how_val, how_tag)
            detail.append((w, how, "", how_val))

        # ---- 牌型奖励与报叫：随胡牌方式——自摸全场付，点炮由点炮者独付 ----
        type_total_by: dict[int, float] = {}
        type_cn_by: dict[int, str] = {}
        for w in winners:
            types = types_by[w]
            type_total = sum(self._hu_type_value(tp) for tp in types)
            type_cn = "+".join(DUSHAN_FAN_CN.get(tp, tp) for tp in sorted(types) if tp != PING_HU)
            type_total_by[w] = type_total
            type_cn_by[w] = type_cn
            if type_total > 0 and payers:
                for q in payers:
                    if q != w:
                        pflow(w, q, type_total, type_cn or "平胡", "牌型")
                detail.append((w, type_cn or "平胡", "牌型", type_total))
            if self.baojiao[w] and cfg.baojiao_bonus > 0 and payers:
                for q in payers:
                    if q != w:
                        pflow(w, q, cfg.baojiao_bonus, "报叫")
                detail.append((w, "报叫", "", cfg.baojiao_bonus))

        # ---- 杠分：明杠只由点杠者付；暗杠/补杠全场每家付；全烧者杠分作废 ----
        for p in range(4):
            if not tenpai[p]:
                continue
            if void_player is not None and p == void_player:
                continue
            for mt, t, src in self.melds[p]:
                if not meld_is_kong(mt):
                    continue
                rel = (src - p) % 4
                who = {1: "下家", 2: "对家", 3: "上家"}.get(rel, "")
                info = {MELD_KONG_CONCEALED: f"暗杠 {tile_cn(t)}",
                        MELD_KONG_ADDED: f"补杠 {tile_cn(t)}"}.get(
                    mt, f"明杠 {tile_cn(t)}" + (f"（{who}点杠）" if who else ""))
                payers_k = [src] if mt == MELD_KONG_EXPOSED \
                    else [q for q in range(4) if q != p]
                for q in payers_k:
                    pflow(p, q, cfg.kong_chickens, "杠分", info)
                detail.append((p, "杠分", info, cfg.kong_chickens))

        # ---- 责任鸡溢价：碰/杠走的鸡，打出者单独赔溢价给碰/杠者 ----
        # （牌面鸡数 碰3/杠4 已计入碰/杠者手中鸡、对所有人有效；
        #   听牌时 AB 之间合计 = 碰冲锋5 / 碰横4 / 杠冲锋9 / 杠横8）
        # 例外：碰/杠者自己未听牌（走包鸡）时不再收溢价——改由包鸡那一步
        # 按「面 + 溢价」赔给打出者，保证 AB 之间合计仍是 4（用户 2026-10-08 例）。
        unt = {q for q in range(4) if not tenpai[q]}
        for claimer, discarder, premium, tag in self._claimed_ji_rows():
            if void_player is not None and claimer == void_player:
                continue        # 全烧者不再获得责任鸡溢价
            if claimer in unt:
                continue        # 未听牌者：溢价并入包鸡反向结算
            pflow(claimer, discarder, premium, "责任鸡", tag)
            detail.append((claimer, "责任鸡", tag, premium))

        # ---- 包鸡/包杠：终局未听牌者对自己**已亮明**的鸡、自己的杠负责（开关控制） ----
        # 「已亮明」= ① 自己打出的鸡（打出去就公开了）＋ ② 副露（碰/杠）里的鸡牌。
        # 手里没亮出来的暗牌鸡**不算**（2026-10-08 用户口径）。
        # 对被自己碰/杠过鸡牌的那家（原打出者），另加该张的溢价：
        # 例：B 碰走 A 的横鸡共 3 张，B 未听牌 → 其他家各得 3，A 得 3+1=4。
        prem_by_discarder: dict[int, list[tuple[str, float]]] = {}
        for cl, dsc, prem, tg in self._claimed_ji_rows():
            prem_by_discarder.setdefault(int(dsc), []).append((tg, float(prem)))
        for p in range(4):
            if tenpai[p]:
                continue
            if cfg.end_baoji:
                ming: list[tuple[str, float]] = list(zip(own_tags[p], own_vals[p]))
                n_yao_m = self.melded_count(p, YAOJI)
                n_oth_m = sum(self.melded_count(p, t)
                              for t in self.ji_tiles if t != YAOJI)
                v_m = (n_yao_m * jin + n_oth_m) * float(cfg.ji_hold)
                if v_m > 0:
                    ming.append((f"副露鸡 {n_yao_m + n_oth_m:g} 张", v_m))
                for tg, val in ming:
                    if val <= 0:
                        continue
                    for q in range(4):
                        if q == p:
                            continue
                        extra, etag = 0.0, ""
                        for ptg, pv in prem_by_discarder.get(q, []):
                            extra += pv
                            etag = ptg
                        tot = val + extra
                        pflow(q, p, tot, "包鸡",
                              tg + (f"+{etag}溢价" if extra else ""), who=p)
                    detail.append((p, "包鸡", tg, val))
            if cfg.end_baogang:
                kongs = sum(1 for mt, _, _ in self.melds[p] if meld_is_kong(mt))
                for _ in range(kongs):
                    for q in range(4):
                        if q != p:
                            pflow(q, p, BAO_KONG, "包杠", who=p)
                    detail.append((p, "包杠", "", BAO_KONG))

        # ---- 两两互减：pair[a][b] = c[a] − c[b]（b 付 a） ----
        # 明细：先列 a 的鸡来源（正），再列 b 的鸡来源（负，抵减），
        # 每行带归属玩家（who），明细行之和恰等于差值
        for a in range(4):
            for b in range(4):
                if a == b:
                    continue
                diff = c[a] - c[b]
                if diff > 0:
                    pair[a][b] += diff
                    rows: list[tuple] = []
                    for label, info, val in items[a]:
                        rows.append((label, info, val, a))
                    for label, info, val in items[b]:
                        if val:
                            rows.append((label, info, -val, b))
                    pair_det[(a, b)] = rows + pair_det.get((a, b), [])

        # ---- 汇总 delta ----
        delta = [0.0] * 4
        for a in range(4):
            for b in range(4):
                delta[a] += pair[a][b] - pair[b][a]

        p_types = types_by[primary]
        fan_cn = "+".join(DUSHAN_FAN_CN.get(tp, tp) for tp in sorted(p_types)) or "平胡"
        fan_total = float(sum(pair[primary]))   # 首位赢家总进账（两两矩阵按行求和）
        # 胡牌张：点炮/抢杠为那张弃牌；自摸为最后摸进的那张（结算页隔开展示）
        if wt_i >= 0:
            wt = wt_i
        elif is_tsumo and getattr(self, "last_draw", None) is not None:
            wt = int(self.last_draw)
        else:
            wt = None
        # 各赢家牌型名/净进账（一炮多响时前端分别展示）
        winners_fan = [{
            "seat": w,
            "fan_cn": "+".join(DUSHAN_FAN_CN.get(tp, tp)
                               for tp in sorted(types_by[w])) or "平胡",
            "fan_type": "+".join(sorted(types_by[w])) or PING_HU,
            "fan": float(sum(pair[w])),
        } for w in winners]
        self.result = {
            "type": "win",
            "winner": primary,
            "winners": list(winners),
            "loser": loser,
            "is_tsumo": is_tsumo,
            "rob_kong": rob_kong,
            "void_player": void_player,
            "win_tile": wt,
            "fan_type": "+".join(sorted(p_types)) or PING_HU,
            "fan_cn": fan_cn,
            "fan": fan_total,
            "total_fan": fan_total,
            "chickens": c,
            # 终局听牌者（座位列表，赢家计入；流水/总览查验用）
            "tenpai": [q for q in range(4) if tenpai[q]],
            "fanji_flip": int(fanji_flip) if fanji_flip is not None else None,
            "fanji_tiles": sorted(int(t) for t in fanji),
            "fanji_gold": bool(gold),
            "jiesuan_fanji": cfg.jiesuan_fanji,
            "kaiju_fanji": cfg.kaiju_fanji,
            "how": how,
            "how_value": float(how_val),
            "type_value": float(type_total_by[primary]),
            "winners_fan": winners_fan,
            "dealer_bonus": 0.0,
            "dou": [],
            "chicken": [d for d in detail if d[1] in CHICKEN_LABELS],
            "detail": detail,
            "pair": pair,
            "pair_detail": [{"a": a, "b": b,
                             "items": [{"label": it[0], "info": it[1], "value": float(it[2]),
                                        "who": int(it[3]) if len(it) > 3 else a}
                                       for it in items_]}
                            for (a, b), items_ in pair_det.items()],
            "wall_left": self.wall_left,
        }
        self._finalize(delta, next_dealer=primary)

    def _settle_huangzhuang(self) -> None:
        """黄牌（荒牌流局）结算：不结算鸡与杠，只做「未听者 → 听牌者」包大牌面。

        口径（用户 2026-10 确认）：
        - **鸡与杠一律不计分**（手中鸡/打出鸡/翻鸡/杠分/责任鸡全部取消），
          也不再做两两鸡数互减；
        - 只在未听牌者与听牌者之间结算：每个未听者向每个听牌者赔
          ``自摸分 + 大牌面分``（按该听牌者可达的最大牌型，以自摸方式结算；
          只有平胡时大牌面为 0，正好等于一个自摸分）；听牌者之间互不结算；
        - ``huang_baoji`` / ``huang_baogang``（默认关）打开时，未听者仍对
          自己打出的鸡/杠向其余三家赔付（保留旧口径开关）。
        """
        cfg: DushanConfig = self.cfg  # type: ignore[assignment]
        detail: list[tuple] = []
        pair = [[0.0] * 4 for _ in range(4)]
        pair_det: dict = {}

        def pflow(a: int, b: int, val: float, label: str, info: str = "",
                  who: int | None = None) -> None:
            """a 收、b 付。``who`` = 该分项归属（责任方口径，默认记在收方 a）。"""
            if val <= 0:
                return
            pair[a][b] += val
            pair_det.setdefault((a, b), []).append(
                (label, info, val, a if who is None else int(who)))

        tenpai = [self._tenpai_flag(p) for p in range(4)]
        baodapai: list[dict] = []

        # ---- 包大牌面：每个未听者向每个听牌者赔「自摸分 + 该听牌者最大牌型分」----
        for p in range(4):
            if not tenpai[p]:
                continue
            tval, tcn = self._max_hu_type(p)
            pay = float(cfg.zi_mo) + float(tval)
            baodapai.append({"seat": p, "type": tcn or "平胡",
                             "type_value": float(tval), "pay": float(pay)})
            if not cfg.huang_baodapai:
                continue
            payers = [q for q in range(4) if not tenpai[q]]
            for q in payers:
                pflow(p, q, pay, "包大牌面", tcn or "平胡", who=q)
            if payers:
                detail.append((p, "包大牌面", tcn or "平胡（自摸鸡）", pay))

        # ---- 可选：包鸡 / 包杠（默认关；打开才结算未听者「已亮明」的鸡与杠）----
        # 口径与胡牌局一致：打出的鸡（含普通鸡）+ 副露里的鸡牌
        if cfg.huang_baoji or cfg.huang_baogang:
            own_vals, own_tags = self._own_ji_after_claims()
            jin_h = float(cfg.jin_ji_multiplier) if YAOJI in self.ji_tiles else 1.0
            for p in range(4):
                if tenpai[p]:
                    continue
                if cfg.huang_baoji:
                    ming: list[tuple[str, float]] = list(zip(own_tags[p], own_vals[p]))
                    n_yao_m = self.melded_count(p, YAOJI)
                    n_oth_m = sum(self.melded_count(p, t)
                                  for t in self.ji_tiles if t != YAOJI)
                    v_m = (n_yao_m * jin_h + n_oth_m) * float(cfg.ji_hold)
                    if v_m > 0:
                        ming.append((f"副露鸡 {n_yao_m + n_oth_m:g} 张", v_m))
                    for tg, v in ming:
                        if v <= 0:
                            continue
                        for q in range(4):
                            if q != p:
                                pflow(q, p, v, "包鸡", tg, who=p)
                        detail.append((p, "包鸡", tg, v))
                if cfg.huang_baogang:
                    kongs = sum(1 for mt, _, _ in self.melds[p] if meld_is_kong(mt))
                    for _ in range(kongs):
                        for q in range(4):
                            if q != p:
                                pflow(q, p, BAO_KONG, "包杠", who=p)
                        detail.append((p, "包杠", "", BAO_KONG))

        delta = [0.0] * 4
        for a in range(4):
            for b in range(4):
                delta[a] += pair[a][b] - pair[b][a]

        self.result = {
            "type": "huangzhuang",
            # 座位列表（听牌者座位；此前误存布尔列表导致前端/统计全部错判）
            "tenpai": [q for q in range(4) if tenpai[q]],
            # 流局不结算鸡（字段保留全 0，避免下游按旧口径展示）
            "chickens": [0.0] * 4,
            # 各听牌者可达最大牌面与「每家赔付额」（结算页展示用）
            "baodapai": baodapai,
            "baodapai_on": bool(cfg.huang_baodapai),
            "cha_jiao": detail,
            "detail": detail,
            "pair": pair,
            "pair_detail": [{"a": a, "b": b,
                             "items": [{"label": it[0], "info": it[1], "value": float(it[2]),
                                        "who": int(it[3]) if len(it) > 3 else a}
                                       for it in items_]}
                            for (a, b), items_ in pair_det.items()],
            "wall_left": 0,
        }
        self._finalize(delta, next_dealer=self.dealer)
