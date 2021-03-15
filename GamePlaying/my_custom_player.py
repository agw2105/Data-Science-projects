#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:56:52 2020

@author: alysonweidmann
"""



import random
from GamePlaying.sample_players import DataPlayer

_WIDTH = 11
_HEIGHT = 9
_SIZE = (_WIDTH + 2) * _HEIGHT - 2


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        
        if state.ply_count < 2:
            #If game is just starting, take the center square if available, or next to it if not
            #try:
            #    self.queue.put(self.center2ind(state))
            #except:
            self.queue.put(random.choice([self.center2ind(state)+1, self.center2ind(state)-1]))

        else:
            self.queue.put(self.alpha_beta(state, 
                                           score_func='defensive', 
                                           depth=2))
        
    @property
    def score_fn(self):
        # store heuristic functions as callables in dictionary
        # can vary scoring mechanism in self.alpha_beta()
        return {'baseline': self.baseline,
                'offensive': self.offensive,
                'defensive': self.defensive,
                'offensive2defensive': self.offensive2defensive,
                'defensive2offensive': self.defensive2offensive,
                'favor_center': self.favor_center,
                'block_opponent': self.block_opponent,
                'heuristic1': self.heuristic1,
                'heuristic2': self.heuristic2,
                'heuristic3': self.heuristic3,
                'heuristic4': self.heuristic4}

    def ratio(self, state):
        area = len(state.liberties(None))
        return state.ply_count / area
    
    def ind2xy(self, ind):
        """ Convert from board index value to xy coordinates

        """
        return (ind % (_WIDTH + 2), ind // (_WIDTH + 2))
   
    def score(self, state):
        #Number own moves available
        own_loc = state.locs[self.player_id]
        own_liberties = state.liberties(own_loc)
        return len(own_liberties)
    
    def opp_score(self, state):
        #Number opponent moves available
        opp_loc = state.locs[1 - self.player_id]
        opp_liberties = state.liberties(opp_loc)
        return len(opp_liberties)
    
    def offensive(self, state):
        #Minimize opponent's available moves at a weighted cost against own
        return self.score(state) - (self.opp_score(state)*2)
    
    def defensive(self, state):
        #Maximize own available moves at weighted cost against opponent's
        return (self.score(state)*2) - self.opp_score(state)
    
    def defensive2offensive(self, state):
        ratio = self.ratio(state)
        if ratio <= 0.5:
            return self.defensive(state)
        else:
            return self.offensive(state)
    
    def offensive2defensive(self, state):
        ratio = self.ratio(state)
        if ratio <= 0.5:
            return self.offensive(state)
        else:
            return self.defensive(state)
        
    def center(self):
        return (_WIDTH // 2, _HEIGHT // 2)
    
    def center2ind(self, state):
        #Convert center xy position to loc index
        center_xy = self.center()
        for i in state.actions():
            if self.ind2xy(i) == center_xy:
                return i
          
    def favor_center(self, state):
        # Get index of center square
        center_xy = self.center()
        
        #Get player positions relative to center
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        
        own_xy = self.ind2xy(own_loc)
        opp_xy = self.ind2xy(opp_loc)
        
        own_distance = abs(own_xy[1] - center_xy[1]) + abs(own_xy[0] - center_xy[0])
        opp_distance = abs(opp_xy[1] - center_xy[1]) + abs(opp_xy[0] - center_xy[0])
        
        #scale to value between -1 and +1, as less important than having more moves than opponent
        return float(own_distance - opp_distance)/10
    
    def block_opponent(self, state):
        # Find opponent moves that are legal moves for the agent and steal them
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc) 
        
        equal_moves = list(set(own_liberties).intersection(opp_liberties))
        
        return self.offensive(state) + len(equal_moves)
    
    def baseline(self, state):
        #my_moves heuristic for baseline comparisons
        return self.score(state) - self.opp_score(state)
    
    def heuristic1(self, state):
        #if players have equal numbers of moves, score to favor the center of the board
        #Otherwise, play offensive2defensive
        if self.score(state) != self.opp_score(state):
            return self.offensive2defensive(state)
        else:
            return self.favor_center(state)
        
    def heuristic2(self, state):
        #if players have equal numbers of moves, score to favor the center of the board
        #Otherwise, play offensive2defensive
        if self.score(state) != self.opp_score(state):
            return self.defensive2offensive(state)
        else:
            return self.favor_center(state)
        
    def heuristic3(self, state):
        #if players have equal numbers of moves, score to favor the center of the board
        #Otherwise, play offensive2defensive
        if self.score(state) != self.opp_score(state):
            return self.offensive(state)
        else:
            return self.favor_center(state)
        
    def heuristic4(self, state):
        #if players have equal numbers of moves, score to favor the center of the board
        #Otherwise, play offensive2defensive
        if self.score(state) != self.opp_score(state):
            return self.defensive(state)
        else:
            return self.favor_center(state)
        
    
    def alpha_beta(self, state, score_func, depth):
       
        def min_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score_fn[score_func](state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                else:
                    beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score_fn[score_func](state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                else:
                    alpha = max(alpha, value)
            return value

        alpha = float("-inf")
        beta = float("+inf")
        depth = depth
        best_score = float("-inf")
        best_move = None

        for a in state.actions():
            v = min_value(state.result(a), alpha, beta, depth - 1)
            alpha = max(alpha, v)
            if v >= best_score:
                best_score = v
                best_move = a
        return best_move
