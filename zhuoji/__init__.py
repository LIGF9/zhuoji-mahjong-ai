"""贵州捉鸡麻将 —— 规则引擎、特征编码与基线 Bot。"""
from .tiles import (  # noqa: F401
    B, DECK_SIZE, NUM_TILE_TYPES, SUITS, TILES_PER_TYPE, T1, W, YAOJI,
    all_tile_names, counts_to_tiles, full_deck, parse_tile, rank_of, suit_of,
    tile_cn, tile_from, tile_name, tiles_to_counts,
)
from .fan import (  # noqa: F401
    FAN_CN, FAN_TABLE, MELD_CHOW, MELD_KONG_ADDED, MELD_KONG_CONCEALED,
    MELD_KONG_EXPOSED, MELD_PONG, classify_fan, is_seven_pairs, is_tenpai,
    is_winning_hand, winning_tiles,
)
from .rules import (  # noqa: F401
    Action, Phase, RulesConfig, ZhuojiGame,
    ANGANG, BUGANG, DISCARD, HU, MINGGANG, PASS, PENG,
)
from .shanten import shanten, useful_tile_score  # noqa: F401

__all__ = [
    "ZhuojiGame", "RulesConfig", "Action", "Phase",
    "shanten", "useful_tile_score",
]
__version__ = "0.1.0"
