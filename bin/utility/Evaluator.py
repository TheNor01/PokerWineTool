class Evaluator:
    def __init__():
        card_order_dict = {"1":2, "2":3, "3":4, "4":5, "5":6, "6":7, "7":8, "9":9, "10":10,"11":11, "12":12, "13":13}

    def check_straight_flush(hand):
        if check_flush(hand) and check_straight(hand):
            return True
        else:
            return False