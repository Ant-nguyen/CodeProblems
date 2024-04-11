
#Leetcode 334

# a way to remember previous best "pair" AND continue forward looking for possible smallest i,j pairing, old 
# j essentially is the representation of the last potential i,j sequence. If something gets pass the i check and 
# is smaller than j, they are the new lower i,j sequence and anything that would have met the condition of 
# the previous i,j,k potential would also be met with the new currest best i,j sequence and any possible k

def increasingTriplet(nums) -> bool:
    i = float("inf")
    j = float("inf")
    for n in nums:
        if n <= i:
            i = n
        elif n <= j:
            j = n
        else:
            return True
    return False

test = [20,100,10,12,5,13]
print(increasingTriplet(test))


