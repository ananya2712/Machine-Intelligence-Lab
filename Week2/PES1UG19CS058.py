from queue import PriorityQueue


def A_star_Traversal(cost, heuristic, start_point, goals):
    path = [] 
    # TODO
    frontier=PriorityQueue()
    visited=[False]*len(cost)
    pathFound=False

    if start_point in goals:
        return [start_point]

    #add the heuristic+cost ,current node and path to priority queue
    frontier.put((0+heuristic[start_point],start_point,[start_point]))

    while(frontier.qsize()):
        path_cost,curr,path=frontier.get()
        visited[curr]=True

        if curr in goals:
            pathFound=True
            break
        
        for i in range(1,len(cost[curr])):
            if visited[i]==True:
                continue
            if visited[i]==False and cost[curr][i]>0:
                new_path  = path + [i]
                new_path_cost = path_cost + cost[curr][i] + heuristic[i] - heuristic[curr]
                frontier.put((new_path_cost,i,new_path))

    return path


def DFS_Traversal(cost,start_point,goals):
    path = []
    frontier = [start_point]
    visited = set()
    pathFound=False

    if start_point in goals:
        return [start_point]

    while len(frontier)>0:
        curr = frontier[-1]
        frontier.pop()

        if curr not in visited:
            path.append(curr)
            visited.add(curr)

        if curr in goals:
            pathFound=True
            break

        neighbour_does_not_exist = 1
        for adjacent_node in range(len(cost)-1,0,-1):
            if adjacent_node not in visited and cost[curr][adjacent_node] > 0:
                frontier.append(adjacent_node)
                neighbour_does_not_exist = 0

        if neighbour_does_not_exist and len(path):
                frontier.append(path[-1])
                children = [i for i in range(len(cost)) if i not in visited and cost[path[-1]][i] > 0 ]
                if len(children) == 0:
                    path.pop()        
                            
    return path 