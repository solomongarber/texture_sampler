import numpy as np
import cv2
import hood_tree

moss=cv2.imread('Moss-1020x610.jpg')
#water=cv2.imread('water.png')
#rnbub=cv2.imread('rnbub.png')
#bio=cv2.imread('biologyThing.jpg')
#sample=rnbub[100:300,250:450]
#sample=water[:200,:200]
#sample=bio[310:510,170:370,:]
sample=moss[200:600,200:600]
sample2=cv2.pyrDown(sample)
sample3=cv2.pyrDown(sample2)
sample4=cv2.pyrDown(sample3)
sample5=cv2.pyrDown(sample4)
t=hood_tree.Tree(sample4,4,(50,50,3),2)

acc=30
while acc>20:
    acc=t.maximization(100)
    t.expectation()
print 'a'
g=t.guess
cv2.imshow('g0',np.uint8(g))

#t.upsample(sample4)
t.change_support(8)
acc=30
while acc>20:
    acc=t.maximization(100)
    t.expectation
print 'b'
g=t.guess
cv2.imshow('g',np.uint8(g))

t.upsample(sample3)
t.change_support(16)
acc=30
while acc>20:
    acc=t.maximization(100)
    t.expectation
print 'b'
g=t.guess
cv2.imshow('gg',np.uint8(g))

t.change_support(8)
acc=30
while acc>20:
    acc=t.maximization(100)
    t.expectation()
print 'c'
g=t.guess
cv2.imshow('g2',np.uint8(g))


t.upsample(sample2)
t.change_support(32)
acc=30
while acc>20:
    acc=t.maximization(20)
    t.expectation()
print 'd'
g=t.guess
cv2.imshow('g3',np.uint8(g))
t.change_support(16)
acc=30
while acc>20:
    acc=t.maximization(100)
    t.expectation()

t.change_support(8)
acc=30
print 'e'
g=t.guess
cv2.imshow('g4',np.uint8(g))
while acc>20:
    acc=t.maximization(100)
    t.expectation()

g=t.guess
cv2.imshow('g5',np.uint8(g))


t.upsample(sample)
t.change_support(32)
acc=30
while acc>20:
    acc=t.maximization(20)
    t.expectation()
print 'f'
g=t.guess
cv2.imshow('g6',np.uint8(g))
t.change_support(16)
acc=30
while acc>20:
    acc=t.maximization(100)
    t.expectation()

t.change_support(8)
acc=30
print 'g'
g=t.guess
cv2.imshow('g7',np.uint8(g))
while acc>20:
    acc=t.maximization(100)
    t.expectation()


cv2.imwrite('./newmoss-400.png',g)

