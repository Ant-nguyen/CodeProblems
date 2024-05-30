
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
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
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


# Leetcode 328: odd even linked list

class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        odd = []
        even =[]
        x = 1
        curr = head
        while curr:
            if x % 2:
                odd.append(curr)
            else:
                even.append(curr)
            x += 1
            curr = curr.next
        for i in range(len(odd)):
            try:           
                odd[i].next = odd[i+1]
            except:
                if even == []:
                    odd[i].next = None
                else:
                    odd[i].next = even[0]
        for i in range(len(even)):
            try:
                even[i].next = even[i+1]
            except:
                even[i].next = None
        return head
    

# Leetcode 2130 Maxmimum Twin Sum of a Linked list
def pairSum(head: ListNode) -> int:
    collection = []
    curr = head
    while curr:
        collection.append(curr.val)
        curr = curr.next
    mid = len(collection)//2
    return max([collection[i] + collection[0-(i+1)] for i in range(mid)])

# Leetcode 2095. Delete the middle node of a linked list
def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if head.next == None:
        return None
    curr = head
    before = head
    mid = head
    x = 1
    while curr:
        curr = curr.next
        x+= 1
        if x%2:
            before = mid
            mid = mid.next
    before.next = mid.next
    return head

# Leetcode 933 : Number of Recent Call
class RecentCounter:

    def __init__(self):
        self.counter = []
        
    def ping(self, t: int) -> int:
        self.counter.append(t)
        while self.counter[0] not in range(t-3000,t+1):
            self.counter.pop(0)
        return len(self.counter)
    

# Leeet code 394: Decode String
# Add each char in the string to the stack, if it is "]", pop from the back until we get to [
# There we look for the number 
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        for char in s:
            if char != "]":
                stack.append(char)
            else:
                sub = ""
                num = ""
                while stack[-1] != "[":
                    #  Note that since we are going backwards from the stack, order is reverse
                    #  hence the pop before the sub
                    sub= stack.pop() + sub
                stack.pop()
                # while Stack is a clever way to make sure the list is not empty.
                while stack and stack[-1].isdigit():
                    num = stack.pop() + num
                stack.append(int(num)*sub)
        return "".join(stack)
    

# Leetcode 649. Dota2 Senate
    def predictPartyVictory(self, senate: str) -> str:
        queue = []
        r= 0
        d = 0
        for s in senate:
            if s == "R":
                r+=1
                queue.append("R")
            else:
                d+=1
                queue.append("D")
        while r and d:
            if queue[0] == "R":
                queue.pop(queue.index("D"))
                d-=1
                queue.append(queue.pop(0))
            else:
                queue.pop(queue.index("R"))
                r-=1
                queue.append(queue.pop(0))
        if r:
            return "Radiant"
        else:
            return "Dire"

# Leetcode 104: Max depth of Binary Tree
# Use recursion.

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        depth_left = self.maxDepth(root.left)
        depth_right = self.maxDepth(root.right)
        return 1+ max(depth_left,depth_right)
    

# LeetCode 872:  Leaf-Similar Trees
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def ends_of_tree(tree):
            if tree == None:
                return ""
            if tree.right == None and tree.left == None:
                return f".{tree.val}."
            left = ends_of_tree(tree.left)
            right = ends_of_tree(tree.right)
            return left + right
        return ends_of_tree(root1) == ends_of_tree(root2)
    

# LeetCode 1448 Count Good Nodes in BT

class TreeNode:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Note the self.counter allowed us to have a static variable to refer to during recursion. 
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        self.counter = 0
        def goodChecker(root: TreeNode,high = float("-inf")):
            if root == None:
                return 
            high = max(root.val,high)
            goodChecker(root.left,high)
            goodChecker(root.right,high)
            if high == root.val:
                self.counter += 1
        goodChecker(root)
        return self.counter
    

# LeetCode 39. Combination sum
# Recursion needed, note the need to copy list
# Every recursion has a split, this ensure combinations ( order does not matter)
# One where it repeats current array and the other where it can not include any in the current array.
# Solution from neetcode. PRACTICE to learn

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []

        def recursion(i,current,total):
            if total == target:
                result.append(current.copy())
                return
            if i >= len(candidates) or total > target:
                return
            current.append(candidates[i])
            recursion(i,current,total+candidates[i])
            current.pop()
            recursion(i+1,current,total)
        recursion(0,[],0)
        return result
        

# LeetCode 437 Path Sum III
# Using a hashmap and dfs rescursion we go through the tree.
# Adding new totals as we go down the tree to the hash map, each key added can be though as a sub path
# We essentailly check how different our current total is to the target, if the difference
# is a previous total it means the current path minus the previous path total would be target
# and that would mean we add to our count.
# We also check everytime if the current total equals the target


from collections import defaultdict
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        if not root:
            return 0 
        self.count, hash = 0,defaultdict(int)
        def dfs(root,total):
            if not root:
                return
            if total == targetSum:
                self.count += 1
            self.count += hash[total-targetSum]
            hash[total] += 1
            if root.left:
                dfs(root.left,total+root.left.val)
            if root.right:
                dfs(root.right,total+root.right.val)
            hash[total] -= 1
        dfs(root,root.val)
        return self.count
    
# LeetCode 236 Lowest Common Ancestor of Binary tree
# Depth first search recurssion to see if a node is any of the targets.
# As we go down we are looking left and right 
# when we find a target, all the previous recusion calls look t see if they ever found
# the other target on their opposing side search. If not we'll return the node target, if
# it does find another target on the other side, we return that node. Recusion loop will
# eventually go back to our root and return our solution

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root == None or root == q or root == p:
            return root
        
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)

        if left and right:
            return root
        return left or right
    

# Leetcode 199. Binary Tree Right side view
# Breadth first search means we use a queue!
# In our solution we use a hashmap and a queue, the queue consist of each node and we keep track of their 
# height. Due to the oddering of how we add to the queue for each height, the furthest right node
# will be last to be added to our hash map. We then take all the heights in our map
# and return it in ascending order.

def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    dicts = {}
    queue = [(root,0)]
    while queue:
        curr = queue.pop(0)
        dicts[curr[1]] = curr[0].val
        if curr[0].left:
            queue.append((curr[0].left,curr[1]+1))
        if curr[0].right:
            queue.append((curr[0].right,curr[1]+1))
    return [dicts[key] for key in sorted(dicts.keys())]

# 1161 Max level Sum of BT

def maxLevelSum( root: Optional[TreeNode]) -> int:
    cur_level = 1
    max_level = 1
    cur_max = float("-inf")
    cur_total = root.val
    queue = [(root,1)]
    while queue:
        curr = queue.pop(0)
        if curr[1] != cur_level:
            if cur_total > cur_max:
                max_level = cur_level
                cur_max = cur_total
            cur_level = curr[1]
            cur_total = 0 
        cur_total += curr[0].val
        if curr[0].left:
            queue.append((curr[0].left,curr[1]+1))
        if curr[0].right:
            queue.append((curr[0].right,curr[1]+1))
    if cur_total > cur_max:
        return cur_level
    return max_level

# Leetcode 841: keys and rooms
def canVisitAllRooms( rooms: List[List[int]]) -> bool:
    hashm = {}
    stack = [0]
    while stack:
        num = stack.pop(0)
        hashm[num] = None
        keys = rooms[num]
        for k in keys:
            if k not in hashm:
                stack.append(k)
    return len(hashm) == len(rooms)

# Leet code 1466: all paths lead to City Zero
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        targets = {0}
        count = 0
        collec = []
        while connections:
            a,b = connections.pop()
            if b in targets:
                targets.add(a)
            elif a in targets:
                count += 1
                targets.add(b)
            else:
                collec.append([a,b])
            if not connections:
                connections =  collec
                collec = []
        return count
    

# Leetcode 66 Plus one:
def plusOne(self, digits: List[int]) -> List[int]:
    digits[-1] += 1
    for ind in reversed(range(len(digits))):
        if digits[ind] >9:
            digits[ind] = 0
            if ind == 0:
                digits = [1] + digits
            else:
                digits[ind-1]+=1
        else:
            break       
    return digits


# Leetcode 300
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        result = [float("inf")]
        for num in nums:
            for i in range(len(result)):
                if num <= result[i]:
                    result[i] = num
                    break
                if i == len(result)-1:
                    result.append(num)
                    break
        return len(result)

# Faster solution that has same logic but uses a bisect to make it much FASTER
from bisect import bisect_left
# bisect returns where a num should inserted to allow for increaseing order.
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = [nums[0]]

        for num in nums[1:]:
            if num > sub[-1]:
                sub.append(num)
            else:
                i = bisect_left(sub, num)
                sub[i] = num

        return len(sub)
    


# Leetcode 547 Number of provinces
def findCircleNum(isConnected: List[List[int]]) -> int:
    unvisited = {n for n in range(len(isConnected))}
    count = 0
    while unvisited:
        stack = [unvisited.pop()]
        while stack:
            cur = isConnected[stack.pop()]
            copy = list(unvisited)
            for n in copy:
                if cur[n]:
                    stack.append(n)
                    unvisited.remove(n)
        count += 1
    return count

# Leet code 1372: Longest zig zag
# Using helper and keeping track of left/right to keep on adding. to the recusion if needed.
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        return max(self.helper(root.right,False,0),self.helper(root.left,True,0))
    
    def helper(self,node,isLeft,depth):
        if not node:
            return depth
        if isLeft:
            depth = max(
                depth,
                self.helper(node.right,False,depth+1),
                self.helper(node.left,True,0))
        else:
            depth = max(
                depth,
                self.helper(node.left,True,depth+1),
                self.helper(node.right,False,0))
        
        return depth


# Leetcode 399

from collections import defaultdict
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        equa = defaultdict(list)
        for ind,eq in enumerate(equations):
            x,y = eq
            equa[x].append((y,values[ind]))
            equa[y].append((x,1/values[ind]))
        
        def bfs(src,target):
            if src not in equa or target not in equa:
                return -1
            q,visit = deque(),{src}
            q.append([src,1])
            while q:
                num,cur = q.popleft()
                if num == target:
                    return cur
                for nei,rate in equa[num]:
                    if nei not in visit:
                        q.append([nei,cur*rate])
                        visit.add(nei)
            return -1
        return [bfs(eq[0],eq[1]) for eq in queries]

# Leetcode 374

# The guess API is already defined for you.
# @param num, your guess
# @return -1 if num is higher than the picked number
#          1 if num is lower than the picked number
#          otherwise return 0
# def guess(num: int) -> int:

class Solution:
    def guessNumber(self, n: int) -> int:
        predict = math.ceil(n/2)
        high = n
        low = 0
        api_guess = guess(predict)
        while api_guess:
            if api_guess == 1:
                low = predict
                predict = math.ceil((high+predict)/2)
            else:
                high = predict
                predict = math.ceil((predict+low)/2)
            api_guess = guess(predict)
        return predict