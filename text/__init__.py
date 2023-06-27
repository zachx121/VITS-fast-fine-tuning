""" from https://github.com/keithito/tacotron """
import logging

from text import cleaners
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, symbols, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  logging.debug(">>> func:text_to_sequence params as follow:")
  logging.debug(f"  [text]: {text}")
  logging.debug(f"  [symbols]: {symbols}")
  logging.debug(f"  [cleaner_names]: {cleaner_names}")
  sequence = []
  symbol_to_id = {s: i for i, s in enumerate(symbols)}
  clean_text = _clean_text(text, cleaner_names)
  logging.debug(f" clean_text: {clean_text}")
  logging.debug(f" length-text:{len(clean_text)}")
  ignored_symbol = []
  for symbol in clean_text:
    if symbol not in symbol_to_id.keys():
      ignored_symbol.append(symbol)
      continue
    symbol_id = symbol_to_id[symbol]
    sequence += [symbol_id]
  logging.debug(f" length-seq:{len(sequence)}")
  logging.debug(" ignored-symbol: '%s'" % ",".join(ignored_symbol))
  return sequence


def cleaned_text_to_sequence(cleaned_text, symbols):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  symbol_to_id = {s: i for i, s in enumerate(symbols)}
  sequence = [symbol_to_id[symbol] for symbol in cleaned_text if symbol in symbol_to_id.keys()]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
