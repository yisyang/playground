from pandas import DataFrame
from random import SystemRandom


def prepare_cards(num_decks=8):
    """
    Prepare decks

    :return: List of shuffled cards as integers, J, Q, K are represented by
             11, 12, 13, respectively.
    """
    sys_rand = SystemRandom()

    # Init 8 decks
    cards = [i for i in range(1, 14)]
    cards = cards * 4 * num_decks
    total_cards = 13 * 4 * num_decks

    # Shuffle cards
    shuffle_order = [sys_rand.randrange(0, total_cards - i) for i in range(total_cards)]
    cards_shuffled = [cards.pop(pos) for pos in shuffle_order]

    # Get them out
    return cards_shuffled


def play_game(cards, num_cards_to_discard=0):
    """
    Play mini-baccarat game according to the rules, and return string as
    'P', 'B', or 'T' depending on who the winner is.

    :param list cards:
    :param int num_cards_to_discard:
    :return: String 'P' if player wins, 'B' if bank wins, or 'T' for tie.
    """
    if num_cards_to_discard:
        cards = cards[num_cards_to_discard:]

    card = cards.pop()
    player_points = card if card < 10 else 0
    card = cards.pop()
    bank_points = card if card < 10 else 0
    card = cards.pop()
    player_points = (card if card < 10 else 0) + player_points
    if player_points >= 10:
        player_points -= 10
    card = cards.pop()
    bank_points = (card if card < 10 else 0) + bank_points
    if bank_points >= 10:
        bank_points -= 10

    # Naturals (8 or 9) get evaluated immediately
    if player_points >= 8 or bank_points >= 8:
        if player_points > bank_points:
            return 'P'
        if bank_points > player_points:
            return 'B'
        return 'T'

    # By default bank does not draw a 3rd card.
    bank_draw = False

    # Player stands on 6 or 7
    if player_points >= 6:
        if bank_points <= 5:
            # Bank will draw on 5 or less if player stands.
            bank_draw = True
    else:
        # Player draws
        card = cards.pop()
        player_points = (card if card < 10 else 0) + player_points
        if player_points > 10:
            player_points -= 10

        if bank_points < 3:
            # Bank will always draw on 0, 1, 2
            bank_draw = True
        elif bank_points == 7:
            bank_draw = False
        elif bank_points == 3:
            bank_draw = card != 8
        elif bank_points == 4:
            bank_draw = card in [2, 3, 4, 5, 6, 7]
        elif bank_points == 5:
            bank_draw = card in [4, 5, 6, 7]
        elif bank_points == 6:
            bank_draw = card in [6, 7]

    if bank_draw:
        card = cards.pop()
        bank_points = (card if card < 10 else 0) + bank_points
        if bank_points >= 10:
            bank_points -= 10

    if player_points > bank_points:
        return 'P'
    if bank_points > player_points:
        return 'B'
    return 'T'


def play_games(num_games, num_decks_in_shoe=8, decks_discarded=2, num_cards_to_discard=0, results_to_track_min=5, results_to_track_max=8):
    """
    Play multiple games and print statistics.

    :param int num_games: Number of games to play.
    :param int num_decks_in_shoe: Number of decks to use.
    :param int decks_discarded: Number of decks to cut, at which time reshuffle will happen.
    :param int num_cards_to_discard: Number of cards to discard per game at start of dealing.
    :param int results_to_track_min: Min length of results to track.
    :param int results_to_track_max: Max length of results to track.
    :return:
    """
    cards = prepare_cards(num_decks_in_shoe)

    # Placeholder to record result stats
    result_signatures = {}
    for i in range(results_to_track_min, results_to_track_max + 1):
        result_signatures[i] = ''
    games_played_for_result_signatures = {}
    player_wins_for_result_signatures = {}
    bank_wins_for_result_signatures = {}
    ties_for_result_signatures = {}

    # Play games
    while num_games > 0:
        num_games -= 1

        # Re-shuffle
        if len(cards) <= decks_discarded * 13 * 4:
            result_signatures = {}
            for i in range(results_to_track_min, results_to_track_max + 1):
                result_signatures[i] = ''
            cards = prepare_cards(num_decks_in_shoe)

        # Play game
        result = play_game(cards, num_cards_to_discard)

        # Record data
        for i in range(results_to_track_min, results_to_track_max + 1):
            if len(result_signatures[i]) == i:
                # Result signature is full and can be used.
                result_signature = result_signatures[i]
                games_played_for_result_signatures[result_signature] = games_played_for_result_signatures.get(
                    result_signature, 0) + 1
                if result == 'P':
                    player_wins_for_result_signatures[result_signature] = player_wins_for_result_signatures.get(
                        result_signature, 0) + 1
                elif result == 'B':
                    bank_wins_for_result_signatures[result_signature] = bank_wins_for_result_signatures.get(
                        result_signature, 0) + 1
                else:
                    ties_for_result_signatures[result_signature] = ties_for_result_signatures.get(
                        result_signature, 0) + 1

            # Update result signature for next iteration
            result_signatures[i] += result
            if len(result_signatures[i]) > i:
                result_signatures[i] = result_signatures[i][1:]

    # Analyze recorded data
    data = {}
    titles = ['Played', 'PWR', 'BWR', 'TR']
    labels = []
    for k in games_played_for_result_signatures:
        games_played = games_played_for_result_signatures.get(k, 0)

        # A small result is going to be statistically insignificant.
        # TODO: Implement formula for variance based on num_games
        # TODO: Auto set results_to_track based on statistical significance
        if games_played < 50000:
            continue

        player_wins = player_wins_for_result_signatures.get(k, 0)
        bank_wins = bank_wins_for_result_signatures.get(k, 0)
        ties = ties_for_result_signatures.get(k, 0)
        player_win_ratio = round(player_wins / games_played * 100, 2)
        bank_win_ratio = round(bank_wins / games_played * 100, 2)
        tie_ratio = round(ties / games_played * 100, 2)

        # You can win at least $1.6 from player or tie on betting $100.
        if player_win_ratio > 50.8 or bank_win_ratio > 53 or tie_ratio > 12.7:
            labels.append(k)
            data[k] = [games_played, player_win_ratio, bank_win_ratio, tie_ratio]

    # Print results
    df = DataFrame(data=data, index=titles)
    print(df)


play_games(500000000, results_to_track_min=7, results_to_track_max=9)

# Result is empty data... nothing with sufficient plays satisfied the meager 101.6/100 requirement.
