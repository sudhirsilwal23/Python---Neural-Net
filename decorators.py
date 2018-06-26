# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:57:02 2018

@author: Sudee
"""

def a_new_decorator(a_func):

    def wrapTheFunction():
        print("before executing a_func()")

        a_func()

        print("after executing a_func()")

    return wrapTheFunction

def a_function_requiring_decoration():
    print("inside a_function_requiring_decoration")

a_function_requiring_decoration()
#outputs: "I am the function which needs some decoration to remove my foul smell"

a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
#now a_function_requiring_decoration is wrapped by wrapTheFunction()

a_function_requiring_decoration()