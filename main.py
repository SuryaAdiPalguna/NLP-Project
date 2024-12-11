import pandas as pd
import pickle
import nltk
import streamlit as st

with open(file="rules/probabilistic_rules.pkl", mode="rb") as file:
  df: pd.DataFrame = pickle.load(file)

grammar_cfg = ""
for prod in df["production"]:
  grammar_cfg += f"{prod}\n"
grammar_cfg = nltk.CFG.fromstring(grammar_cfg)
grammar_cfg._start = nltk.Nonterminal("K")
parser_cfg = nltk.ChartParser(grammar_cfg)

grammar_pcfg = ""
for prod, prob in zip(df["production"], df["probability"]):
  grammar_pcfg += f"{prod} [{prob}]\n"
grammar_pcfg = nltk.PCFG.fromstring(grammar_pcfg)
grammar_pcfg._start = nltk.Nonterminal("K")
parser_pcfg = nltk.ViterbiParser(grammar_pcfg)

sentence = st.text_input("Enter your Balinese sentence: ")
button = st.button("Parse")

if button:
  sentence = sentence.lower().split()
  trees_pcfg = list(parser_pcfg.parse(sentence))
  trees_cfg = list(parser_cfg.parse(sentence))

  st.write("Hasil Probabilistic CFG:")
  if len(trees_pcfg) == 0:
    st.info("This sentence haven't parsing result")
  for t in trees_pcfg:
    svg = t._repr_svg_()
    st.write(svg, unsafe_allow_html=True)
    st.info(f"Score: {t.prob()}")

  st.write("Hasil CFG:")
  if len(trees_cfg) == 0:
    st.info("This sentence haven't parsing result")
  for t in trees_cfg:
    svg = t._repr_svg_()
    st.write(svg, unsafe_allow_html=True)
    