#ifndef _MAIN_H_
#define _MAIN_H_

#include <iostream>
#include <vector>
#include <stdio.h>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fst/fstlib.h>

#include <cblas.h>

#include <alsa/asoundlib.h>

template <typename Token>
struct ForwardLink {
  Token *next_tok;  // the next token [or NULL if represents final-state]
  int ilabel;  // ilabel on arc
  int olabel;  // olabel on arc
  float graph_cost;  // graph cost of traversing arc (contains LM, etc.)
  float acoustic_cost;  // acoustic cost (pre-scaled) of traversing arc
  ForwardLink *next;  // next in singly-linked list of forward arcs (arcs
                      // in the state-level lattice) from a token.
  inline ForwardLink(Token *next_tok, int ilabel, int olabel,
                     float graph_cost, float acoustic_cost,
                     ForwardLink *next):
      next_tok(next_tok), ilabel(ilabel), olabel(olabel),
      graph_cost(graph_cost), acoustic_cost(acoustic_cost),
      next(next) { }
};

struct StdToken {
  using ForwardLinkT = ForwardLink<StdToken>;
  using Token = StdToken;

  float tot_cost;

  float extra_cost;

  ForwardLinkT *links;
  Token *next;
  inline void SetBackpointer (Token *backpointer) { }
  inline StdToken(float tot_cost, float extra_cost, ForwardLinkT *links,
                  Token *next, Token *backpointer):
      tot_cost(tot_cost), extra_cost(extra_cost), links(links), next(next) { }
};

struct TokenList {
  StdToken *toks;
  bool must_prune_forward_links;
  bool must_prune_tokens;
  TokenList(): toks(NULL), must_prune_forward_links(true),
              must_prune_tokens(true) { }
};
#endif