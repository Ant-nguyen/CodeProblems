
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


# LeetCode 26

# changes need to happen in place
# esentially just look for each unique value to add to the next earliest position on the list
def removeDuplicates(nums) -> int:
    i = 0
    for j in range(1,len(nums)):
        if nums[i] != nums[j]:
            i+= 1
            nums[i]=nums[j]
    return i+1


# Leet code 128.  Longest Consecutive Sequence
def longestConsecutive( nums) -> int:
    # edge case catch
    if len(nums)==0:
        return 0
    sorts = sorted(nums)
    hash = {num: 1 for num in nums}
    for i,num in enumerate(sorts):
        #Check to see if previous number is sequential
        if sorts[i-1] == num-1:
            # take the previous numbers sequential total and add to current
            hash[num] += hash[num-1]
    return max(hash.values())


# Leet Code 11: Container With Most Water
def func (arr):
    l = 0
    r = len(arr)-1
    w = r - l
    low_height = min(arr[l],arr[r])
    max_area = low_height * w
    while w != 0:
        # left?
        if low_height == arr[l]:
            l += 1
            w -= 1
            low_height = min(arr[l],arr[r])
            area = low_height*w
            if area > max_area:
                max_area = area
        # right
        else:
            r -= 1
            w -= 1
            low_height = min(arr[l],arr[r])
            area = low_height*w
            if area > max_area:
                max_area = area
    return max_area


# LeetCode 1679. Max Number of k-sum Pairs

# Brute force way using HashTable:

# class Solution:
#     def maxOperations(self, nums: List[int], k: int) -> int:
#         hash = {}
#         count = 0
#         for ind,num in enumerate(nums):
#             if num in hash:
#                 hash[num].pop(0)
#                 if len(hash[num]) == 0:
#                     del hash[num]
#                 count += 1
#             else:
#                 if k-num in hash:
#                     hash[k-num].append(ind)
#                 else:
#                     hash[k-num] = [ind]
#         return count

# Solution using two Pointers:
def maxOperations(nums, k: int) -> int:
    nums.sort()
    left,right,count = 0,len(nums)-1,0
    while left < right:
        if nums[left] + nums[right] == k:
            count+= 1
            left += 1
            right -= 1
            continue
        elif nums[left] + nums[right] > k:
            right -= 1
        else:
            left += 1
    return count



# LeetCode: 1456. Maximum Number of Vowels in a Substring of Given Length
# Sliding window
def maxVowels(self, s: str, k: int) -> int:
    vowels = {"a","e","i","o","u"}
    count = sum([1 for n in s[0:k] if n in vowels])
    max = count
    # starting at k essentially makes i the end of the window and i-k the front!
    for i in range(k,len(s)):
        if count == k:
            return k
        if s[i] in vowels:
            count += 1
        if s[i-k] in vowels:
            count -= 1
        if count > max:
            max = count
    return max


# leetcode 1004 : Max Consercutive ones III
# Window/ two pointer solution

def longestOnes(self, nums: List[int], k: int) -> int:
    l= maxCon = 0
    for n in range(len(nums)):
        # Cheeky way of condesing and if statement and assiment that works because were dealing with
        # 1 and 0s
        k -= 1-nums[n]
        if k <0:
            if nums[l] == 0:
                k+=1
            l+=1
        else:
            maxCon = max(maxCon,n-l+1)
    return maxCon


# Leetcode 1493: Longest subarray of 1's after deleting
# Window solution? Change window based on it Zero is found.
def longestSubarray(self, nums: List[int]) -> int:
    l=r=0
    maxTotal = 0
    zero= False
    for num in nums:
        r+= num
        if not num:
            zero =True
            l = r
            r = 0 
        if r+l > maxTotal:
            maxTotal = r+l

    if zero:
        return maxTotal 
    else:
        return len(nums)-1
    

# Leetcode 2390: removing Stars from a String:
# Use Stack...
class Solution:
    def removeStars(self, s: str) -> str:
        stack = []
        for let in s:
            if let == "*":
                stack.pop()
            else:
                stack.append(let)
        return "".join(stack)
    

# Leetcode 206: Reverse Linked list:
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev,curr = None,head
        while curr:
            temp = prev
            prev = curr
            curr = curr.next
            prev.next = temp
        return prev
    



# leetcode 1657: Determine if two strings are close:
import collections

def closeStrings(self, word1: str, word2: str) -> bool:
    if len(word1) != len(word2):
        return False
    one = collections.Counter(word1)
    two = collections.Counter(word2)
    
    first = sorted(one.values())
    second = sorted(two.values())
    return first == second and set(one.keys()) == set(two.keys())


# Leetcode 2352: Equal Row and col Pairs

def equalPairs(grid) -> int:
    count = 0
    hash = {}
    # Make rows
    for row in grid:
        if tuple(row) not in hash:
            hash[tuple(row)] = 1
        else:
            hash[tuple(row)] += 1
    for i in range(len(grid)):
        # Make columns 
        col = [grid[j][i] for j in range(len(grid))]
        # Check if they are there. 
        if tuple(col) in hash:
            count += hash[tuple(col)]
    return count
