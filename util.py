#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mathias.mueller@uzh.ch

import re

_emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)


def remove_emojis(string_):
    """
    Remove most emojis from text.
    """
    return _emoji_pattern.sub(r'', string_)
