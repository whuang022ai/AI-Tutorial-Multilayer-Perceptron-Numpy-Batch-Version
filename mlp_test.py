# -*- coding: utf-8 -*-
#
#  Testing Xor pre-trained model.
#  @auth whuang022ai
#

import mlp 

if __name__ == "__main__":
    mlp=mlp.MLP(0,0,0,0)
    mlp.load_model('xor')
    mlp.test_forward()