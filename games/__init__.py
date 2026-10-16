"""Game implementations for the AlphaZero engine.

Each game module implements the uniform API consumed through
``game_api`` (see ``game_api.py``) and must export at least:

- name, board_h, board_w, num_planes, num_actions, policy_channels
  (dimension attributes)
- new_state(start), apply(state, move), legal_moves(state),
  state_ply(state), is_terminal(state), terminal_value(state),
  encode(state), mask(state), to_index(state, move),
  index_to_move(state, idx), move_key(move), move_str(move),
  describe(state), player_parity(state), player_sign(state),
  final_result(state, move_count, max_game_length, **kwargs),
  reference_move(state, **kwargs)
"""