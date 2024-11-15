#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import collections
import csv
import itertools
import json
import re
import os



__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "1.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


ROLE_TEACHER = "teacher"
ROLE_STUDENT = "student"

TASK = 'TSCC'


def _make_dialogic_pairs(role_first=None):
    cache = dict(counter=0)

    def func(role):
        if role == role_first:
            cache['counter'] += 1
            return cache['counter']
        else:
            return cache['counter']

    return func


def _concat_turns(turns, use_edits=False):
    turns = list(turns)
    if use_edits:
        turn_sq = '\n'.join(t.clean(t.edited) if t.edited else t.clean(
            t.anonymised) for t in turns)
    else:
        turn_sq = '\n'.join(t.clean(t.anonymised) for t in turns)
    return turn_sq


class Chat(object):

    def __init__(self, turns) -> None:
        super().__init__()
        self._turns = collections.OrderedDict({t.number: t for t in turns})

    @property
    def turns(self):
        return self._turns.values()

    def get_turn(self, number):
        return self._turns.get(number)

    @classmethod
    def from_tsv(cls, filename):
        with open(filename) as fh:
            reader = csv.reader(fh, delimiter='\t')
            next(reader, None)  # read header
            turns = [Turn.from_row(i, line) for i, line in enumerate(reader)]
        return cls(turns)


class Turn(object):

    NA_VAL = "NA"
    ROLE_TEACHER = ROLE_TEACHER
    ROLE_STUDENT = ROLE_STUDENT

    def __init__(self,
                 line_idx,
                 timestamp,
                 user_id,
                 role,
                 turn_number,
                 anonymised,
                 edited,
                 responding_to=None,
                 sequence=None,
                 seq_type=None,
                 focus=None,
                 resource=None,
                 assessment=None) -> None:
        super().__init__()
        self.line_idx = line_idx
        self.timestamp = timestamp
        self.user_id = user_id
        self.role = role
        self.number = int(turn_number)
        self.anonymised = anonymised
        self.edited = edited
        self.responding_to = responding_to
        self.sequence = sequence
        self.seq_type = seq_type
        self.focus = focus
        self.resource = resource
        self.assessment = assessment

    @property
    def text(self):
        return self.anonymised

    @classmethod
    def from_row(cls, line_idx, cols):
        return cls(line_idx,
                   *map(lambda x: None if x == cls.NA_VAL else x, cols))

    @staticmethod
    def clean(turn_str):
        turn_new = re.sub(r"<([A-Z]+)([ '][A-Z ']+)?>",
                          lambda m: m.group(1).lower(), turn_str)
        turn_new = turn_new
        return turn_new


