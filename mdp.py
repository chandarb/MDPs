from parameters import *

class MDP(object):

    def __init__(self, T=30, method="linear", features="random"):
        """Construct the space and run value iteration.
        T is the number of iterations for value iteration.
        If method is "linear" does linear interpolation over the features to 
        estimate the value function. Otherwise it uses a random forest regression.
        If features is "random" it draws random features using the conditional 
        normal distribution. Otherwise it uses deterministic features based
        on the position and orientation of the point relative to the terminal states"""
        # mapping of terminal states to their reward
        self.terminals = {}
        # centers of each terminal state
        self.centers = []
        self.center_vals = {}
        # circle objects
        self.ys = []
        # sampled points
        self.sample_states = []

        self.__draw_circles()
        self.__draw_sample()

        self.thetas = []
        if features=="random":
            self.thetas = self.draw_y_th(self.sample_states)
        else:
            self.thetas = self.find_all_thetas(self.sample_states)

        self.v_t, self.policy = self.value_iteration(T, method=method, features=features)



    def __draw_circles(self):
        good_s = 0
        # create the terminal states
        # find non-overlapping circles in the space
        while True:
            xi = np.random.uniform(xmin + c_size, xmax - c_size)
            yi = np.random.uniform(ymin + c_size, ymax - c_size)
            # create circle at (xi, yi)
            circlei = Point(xi, yi).buffer(c_size)
            overlap = False
            # see if it overlaps with a circle already created
            for c in self.terminals:
                if c.intersects(circlei):
                    overlap = True
                    break
            if overlap:
                continue

            if good_s < num_good:
                # make the circle a good terminal state
                self.terminals[circlei] = 10
                self.ys.append(circlei)
                good_s += 1
                self.centers.append((xi, yi))
                self.center_vals[(xi,yi)] = 10
            else:
                # otherwise it's a bad terminal state
                self.terminals[circlei] = -10
                self.ys.append(circlei)
                self.centers.append((xi, yi))
                self.center_vals[(xi,yi)] = -10
            if (len(self.terminals) == total):
                break
        return

    def draw_point(self):
        """sample a point in the space"""
        while True:
            xi = np.random.uniform(xmin, xmax)
            yi = np.random.uniform(ymin, ymax)
            inside = self.in_terminal((xi,yi))
            if inside:
                continue
            else:
                break
        return (xi,yi)

    def __draw_sample(self):
        """sample points over the space
        they shouldn't lie in the terminal regions"""
        for i in range(N):
            self.sample_states.append(self.draw_point())
        return


    def draw_y_th(self, sample):
        """Associate each point in the sample with a terminal state and get features"""
        n = len(sample)
        # unnormalized probabilities of each class for each point
        probs = np.zeros((n, total))
        for k in range(len(self.centers)):
            mu_k = self.centers[k]
            for i in range(n):
                probs[i, k] = norm_pdf_multivariate(np.array(sample[i]), np.array(mu_k), cov_s)

        # normalizing constants
        const = np.transpose(np.matrix(np.sum(probs, axis=1)))
        # create a matrix of the normalizing constants
        d_mat = np.matrix(const)
        for i in range(k):
            d_mat = np.concatenate((d_mat, const), axis=1)

        # normalize the probabilities to get the 
        # distribution of Y | s in each row
        probs = np.divide(probs, d_mat)

        ks = []

        # draw the class for each sample point
        for i in range(np.shape(probs)[0]):
            p = probs[i,].tolist()[0]
            draw = np.random.multinomial(1, p).tolist()
            ind = draw.index(1)
            ks.append(ind)

        thetas = []
        # draw features for each point s
        for i in range(n):
            k = ks[i]
            s = sample[i]
            s_mean = self.centers[k]
            theta_k = theta_mu[k]
            c_mean = theta_k + np.dot(scale, np.array(s)-np.array(s_mean))
            # draw from the conditional multivariate normal
            thetas.append(np.random.multivariate_normal(c_mean, c_cov))
        return thetas

    def find_thetas(self, p):
        """Finds deterministic feature vectors.
        distance to closest good state, closest bad state
        orientation to closest good state, orientation to closest bad state"""
        cl_good = (0,0)
        cl_bad = (0,0)
        dis_g = 100
        dis_b = 100
        dis = 100
        for c in self.centers:
            c_dis = l2norm(p, c)
            if self.center_vals[c] > 10:
                if c_dis < dis_g:
                    dis_g = c_dis
                    cl_good = c
            else:
                if c_dis < dis_b:
                    dis_b = c_dis
                    cl_bad = c
        return (dis_g, dis_b, angle(p, cl_good), angle(p, cl_bad))
        
    def find_all_thetas(self, points):
        """Finds deterministic feature vectors for all points in the sample"""
        return map(lambda x: self.find_thetas(x), points)

    def get_rewards(self, sp):
        "get the reward for being in a state"
        rewards = [0 for i in range(s_fine)]
        for j in range(s_fine):
            s = sp[j]
            p = Point(s[0], s[1])
            # assign reward if in terminal state
            for y in self.ys:
                if y.contains(p):
                    rewards[j] = self.terminals[y]
                    break
        return np.array(rewards)

    def value_iteration(self, T, method="linear", features="random"):
        "run value iteration over each of the sample points"
        # initial value vector
        v_ti = np.zeros((N))
        v_t1 = np.zeros((N))
        # initial policy function
        policy = [0 for i in range(N)]
        # feature matrix for value function regression
        x_mat = np.concatenate((self.sample_states, self.thetas), axis=1)
        for t in range(T):
            # linear interpolation of value function
            if method=="linear":
                clf = linear_model.LinearRegression()
                clf.fit(x_mat, v_ti)
                coeffs = clf.coef_
                intercept = clf.intercept_
            # random forest interpolation
            elif method=="forest":
                clf = RandomForestRegressor(max_depth=d * total)
                clf.fit(x_mat, v_ti)

            for i in range(N):
                p = self.sample_states[i]
                # sample across possible actions
                actions = sample_circle(p, dist, fine)
                max_val = -1000
                max_action = (0, 0)
                # find the action that maximizes the value function
                for a in actions:
                    # draw possible states since there's noisy motion
                    s_p = np.random.multivariate_normal(a, noise_mat, s_fine)
                    # draw features for these states
                    if features=="random":
                        sp_t = self.draw_y_th(s_p)
                    else:
                        sp_t = self.find_all_thetas(s_p)
                    feature_mat = np.concatenate((s_p, sp_t), axis=1)
                    # discounted future value
                    if method=="linear":
                        vals = np.transpose((np.dot(feature_mat, coeffs) + intercept) * gamma)
                    elif method=="forest":
                        vals = clf.predict(feature_mat) * gamma
                    # density of each state
                    probs = []
                    for st in s_p:
                        probs.append(norm_pdf_multivariate(np.array(st), np.array(a), noise_mat))
                    probs = np.array(probs)
                    const = np.sum(probs)
                    # rewards for each state
                    rewards = np.transpose(self.get_rewards(s_p))
                    u_val = np.dot(probs, rewards) + np.dot(probs, vals)
                    # empirical expectation of the reward for an action
                    sp_val = u_val / const
                    # find the maximum possible value
                    if sp_val > max_val:
                        max_val = sp_val
                        max_action = np.array(a) - np.array(p)
                # update the value and policy for a sample point
                v_t1[i] = max_val
                policy[i] = max_action
            v_ti = list(v_t1)
        return v_ti, policy


    def in_terminal(self, p):
        """Check whether a point is in a terminal state"""
        po = Point(p[0], p[1])
        inside = False
        for y in self.ys:
            if y.contains(po):
                inside = True
                break
        return inside

    def find_path(self, p):
        """Try to find an optimal set of actions from some starting point in the space.
        Follows the actions of the closest sampled point until
        a terminal state is reached or the length of the path is 
        too long."""
        # start point in the path
        path = [p]
        actions = []
        a_centers = []
        while not in_terminal(p):
            list_p = [p for i in range(N)]
            # get distance to every sampled point
            distances = map(l2norm, self.sample_states, list_p)
            # find the closest sampled point
            closest = distances.index(min(distances))
            # follow the same policy as for the closest point
            action = self.policy[closest]
            # expected new location
            a_center = np.array(action) + np.array(p)
            # new location after noise in motion
            p = tuple(np.random.multivariate_normal(a_center, noise_mat, 1).tolist()[0])
            actions.append(action)
            a_centers.append(a_center)
            # new path
            path.append(p)
            # path is too long--stop
            if len(path) > 50:
                break
        return path, actions, a_centers

    def find_path_optim(self, p, method="linear"):
        """Try to find an optimal set of actions from some starting point in the space.
        Optimizes over the set of possible actions at the point to take the next step in the
        path."""
        path = [p]
        actions = []
        a_centers = []
        x_mat = np.concatenate((self.sample_states, self.thetas), axis=1)
        if method=="linear":
            clf = linear_model.LinearRegression()
            clf.fit(x_mat, v_t)
            coeffs = clf.coef_
            intercept = clf.intercept_
        elif method=="forest":
            clf = RandomForestRegressor(max_depth=d * total)
            clf.fit(x_mat, v_t)
        while not in_terminal(p):
            list_p = [p for i in range(N)]
            actions = sample_circle(p, dist, fine)
            max_val = -1000
            max_action = (0, 0)
            # find the action that maximizes the value function
            for a in actions:
                # draw possible states since there's noisy motion
                s_p = np.random.multivariate_normal(a, noise_mat, s_fine)
                # draw features for these states
                sp_t = self.draw_y_th(s_p)
                feature_mat = np.concatenate((s_p, sp_t), axis=1)
                # discounted future value
                vals = clf.predict(feature_mat) * gamma
                # discounted future value
                if method=="linear":
                    vals = np.transpose((np.dot(feature_mat, coeffs) + intercept) * gamma)
                elif method=="forest":
                    vals = clf.predict(feature_mat) * gamma
                # density of each state
                probs = []
                for st in s_p:
                    probs.append(norm_pdf_multivariate(np.array(st), np.array(a), noise_mat))
                probs = np.array(probs)
                const = np.sum(probs)
                # rewards for each state
                rewards = np.transpose(self.get_rewards(s_p))
                u_val = np.dot(probs, rewards) + np.dot(probs, vals)
                # empirical expectation of the reward for an action
                sp_val = u_val / const
                # find the maximum possible value
                if sp_val > max_val:
                    max_val = sp_val
                    max_action = np.array(a) - np.array(p)
            # expected new position after taking optimal action
            a_center = np.array(max_action) + np.array(p)
            # true position after taking action
            p = tuple(np.random.multivariate_normal(a_center, noise_mat, 1).tolist()[0])
            actions.append(max_action)
            a_centers.append(a_center)
            path.append(p)
            # break if path too long
            if len(path) > 50:
                break
        return path, actions, a_centers

    def plot_space(self, plot_path=False, method="closest"):
        """Plot the space and policy. 

        If plot_path is True plot the path for some random 
        point in the space. If method is "closest" follow the 
        policy of the closest sampled point. Otherwise reoptimize 
        over the set of possilve actions to solve for the path."""
        # create the terminal states
        c_plot = []
        for c in self.centers:
            # bad states are magenta
            color = 'm'
            # good states are green
            if self.center_vals[c] > 0:
                color = 'g'
            c_plot.append(plt.Circle(c, c_size, color=color))

        fig = plt.gcf()
        # add the terminal states to the plot
        for c in c_plot:
            fig.gca().add_artist(c)

        # get all the sampled points
        xy = zip(*self.sample_states)
        xs0 = list(xy[0])
        ys0 = list(xy[1])

        # get the policy for each sampled point
        dxdy = zip(*self.policy)
        dx = list(dxdy[0])
        dy = list(dxdy[1])

        x1 = np.array(xs0) + np.array(dx)
        y1 = np.array(ys0) + np.array(dy)

        # construct lines for the policy
        linesx = []
        linesy = []
        for i in range(N):
            linesx.append(xs0[i])
            linesx.append(x1[i])
            linesx.append(None)
            linesy.append(ys0[i])
            linesy.append(y1[i])
            linesy.append(None)

        if plot_path:
            # plot the optimal path for some sampled point
            sample_point = M.draw_point()
            path=[]
            actions=[]
            action_centers=[]
            if method=="closest":
                path, actions, action_centers = M.find_path(sample_point)
            else:
                path, actions, action_centers = M.find_path(sample_point)
            path_xy = zip(*path)
            path_x = list(path_xy[0])
            path_y = list(path_xy[1])

            path_lx = []
            path_ly = []
            for i in range(len(path_x) - 1):
                path_lx.append(path_x[i])
                path_lx.append(path_x[i+1])
                path_lx.append(None)
                path_ly.append(path_y[i])
                path_ly.append(path_y[i+1])
                path_ly.append(None)
            plt.plot(path_lx, path_ly, 'y')

        plt.plot(linesx, linesy)
        plt.plot(xs0, ys0, 'ro')

        plt.show()
        return



def angle(p1, p2):
    """Compute the angle between two points"""
    diff = np.array(p1) - np.array(p2)
    dy = diff[1]
    dx = diff[0]
    return(np.arctan(dy/dx))

def norm_pdf_multivariate(x, mu, sigma):
    """pdf of multivariate normal"""
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


def x_coord(num, radius, center): 
    """get the x coordinate for some (r, theta)"""
    return (math.cos(num) * radius) + center

def y_coord(num, radius, center):
    """get the y coordinate for some (r, theta)"""
    return (math.sin(num) * radius) + center

def sample_circle(point, radius, fineness):
    """sample points from the boudnary of a circle"""
    x_0 = point[0]
    y_0 = point[1]
    # map the interval to the circle
    s = np.random.uniform(0, 2 * math.pi, fineness)
    r = [radius for i in range(fineness)]
    x_cen = [x_0 for i in range(fineness)]
    y_cen = [y_0 for i in range(fineness)]
    x = map(x_coord, s, r, x_cen)
    y = map(y_coord, s, r, y_cen)
    return zip(x, y)


def l2norm(p1, p2):
    """Get the squared L2 norm between two points"""
    coords1 = np.array(p1)
    coords2 = np.array(p2)
    return sum((coords1 - coords2)**2)


