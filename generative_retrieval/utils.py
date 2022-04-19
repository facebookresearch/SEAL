import torch
from torch import nn



stopwords = set(
    [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 28, 30, 31, 32, 33, 34,
    37, 38, 39, 40, 41, 42, 45, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62,
    63, 64, 66, 69, 70, 71, 77, 79, 81, 83, 84, 85, 87, 88, 89, 91, 95, 96, 97, 98, 99,
    103, 106, 109, 110, 114, 117, 122, 123, 125, 127, 129, 136, 137, 141, 142, 143, 144,
    145, 147, 148, 149, 150, 152, 159, 160, 162, 166, 167, 172, 178, 182, 197, 208, 209,
    211, 215, 218, 222, 223, 227, 252, 255, 256, 258, 259, 264, 276, 280, 286, 287, 308,
    318, 326, 345, 349, 350, 351, 367, 370, 374, 384, 385, 399, 404, 407, 440, 454, 456,
    473, 475, 497, 519, 520, 524, 572, 579, 590, 596, 598, 608, 616, 617, 630, 653, 660,
    683, 769, 832, 854, 870, 874, 901, 938, 939, 965, 978, 993, 1003, 1021, 1065, 1216,
    1223, 1235, 1308, 1336, 1398, 1405, 1423, 1456, 1464, 1491, 1495, 1525, 1534, 1541,
    1590, 1599, 1705, 1740, 1793, 1801, 1832, 1868, 1892, 1918, 1936, 1941, 1944, 1979,
    1993, 2025, 2096, 2185, 2220, 2246, 2282, 2290, 2306, 2486, 2512, 2548, 2612, 2615,
    2661, 2667, 2808, 2864, 3001, 3047, 3066, 3105, 3128, 3139, 3224, 3243, 3263, 3326,
    3394, 3486, 3559, 3703, 3779, 3842, 3945, 4028, 4041, 4248, 4288, 4395, 4421, 4820,
    4979, 4995, 5030, 5053, 5089, 5102, 5121, 5365, 5570, 5598, 5818, 5844, 6015, 6233,
    6278, 6319, 6362, 6532, 6553, 6567, 6834, 6871, 7029, 7301, 7574, 7698, 8127, 8228,
    8374, 8901, 8981, 9012, 9131, 9174, 9443, 10284, 10414, 10540, 10616, 10652, 10786,
    10978, 11974, 12050, 12135, 12178, 12341, 12389, 12471, 12655, 12925, 13331, 13387,
    13464, 13910, 14003, 14010, 14279, 14314, 14662, 15157, 15446, 16005, 16536, 16897,
    17345, 17346, 17717, 17754, 18212, 18258, 18342, 18630, 18966, 19385, 19935, 19981,
    20060, 20311, 20343, 20685, 20693, 21097, 21688, 22008, 22062, 23367, 24394, 24975,
    24980, 25133, 26021, 26421, 26817, 27409, 27789, 28595, 28842, 29856, 29892, 29919,
    30518, 30536, 30540, 30857, 31455, 31940, 31954, 31963, 32060, 32431, 32882, 34157,
    34290, 34912, 35420, 35669, 36698, 37051, 37350, 38374, 38642, 39269, 39789, 39973,
    40375, 40930, 41812, 41814, 42068, 43216, 43570, 44471, 45893]
)


def _remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def load_state_dict_from_lightning_checkpoint(model, path):
    state_dict = torch.load(path, map_location='cpu')
    # state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    # for key in ['shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']:
    #         state_dict[key] = torch.cat([state_dict[key], torch.zeros_like(state_dict[key][:1])], 0)
    # _remove_ignore_keys_(state_dict)
    # if hasattr(model, "lm_head"):
    #     model.lm_head = _make_linear_from_emb(model.model.shared)
    model.load_state_dict(state_dict)


def load_state_dict_from_fairseq_checkpoint(model, path):
    state_dict = torch.load(path, map_location='cpu')['model']
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    for key in ['shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']:
            state_dict[key] = torch.cat([state_dict[key], torch.zeros_like(state_dict[key][:1])], 0)
    _remove_ignore_keys_(state_dict)
    if hasattr(model, "lm_head"):
        model.lm_head = _make_linear_from_emb(model.model.shared)
    model.model.load_state_dict(state_dict)