def iup_segment(coords, rc1, rd1, rc2, rd2):
	# rc1 = reference coord 1
	# rd1 = reference delta 1
	out_arrays = [None, None]
	for j in 0,1:
		out_arrays[j] = out = []
		x1, x2, d1, d2 = rc1[j], rc2[j], rd1[j], rd2[j]


		if x1 == x2:
			n = len(coords)
			if d1 == d2:
				out.extend([d1]*n)
			else:
				out.extend([0]*n)
			continue

		if x1 > x2:
			x1, x2 = x2, x1
			d1, d2 = d2, d1

		# x1 < x2
		scale = (d2 - d1) / (x2 - x1)
		for pair in coords:
			x = pair[j]

			if x <= x1:
				d = d1
			elif x >= x2:
				d = d2
			else:
				# Interpolate
				d = d1 + (x - x1) * scale

			out.append(d)

	return zip(*out_arrays)

def iup_contour(delta, coords):
	assert len(delta) == len(coords)
	if None not in delta:
		return delta

	n = len(delta)
	# indices of points with explicit deltas
	indices = [i for i,v in enumerate(delta) if v is not None]
	if not indices:
		# All deltas are None.  Return 0,0 for all.
		return [(0,0)]*n

	out = []
	it = iter(indices)
	start = next(it)
	if start != 0:
		# Initial segment that wraps around
		i1, i2, ri1, ri2 = 0, start, start, indices[-1]
		out.extend(iup_segment(coords[i1:i2], coords[ri1], delta[ri1], coords[ri2], delta[ri2]))
	out.append(delta[start])
	for end in it:
		if end - start > 1:
			i1, i2, ri1, ri2 = start+1, end, start, end
			out.extend(iup_segment(coords[i1:i2], coords[ri1], delta[ri1], coords[ri2], delta[ri2]))
		out.append(delta[end])
		start = end
	if start != n-1:
		# Final segment that wraps around
		i1, i2, ri1, ri2 = start+1, n, start, indices[0]
		out.extend(iup_segment(coords[i1:i2], coords[ri1], delta[ri1], coords[ri2], delta[ri2]))

	assert len(delta) == len(out), (len(delta), len(out))
	return out

def iup_delta(delta, coords, ends):
	assert sorted(ends) == ends and len(coords) == (ends[-1]+1 if ends else 0) + 4
	n = len(coords)
	ends = ends + [n-4, n-3, n-2, n-1]
	out = []
	start = 0
	for end in ends:
		end += 1
		contour = iup_contour(delta[start:end], coords[start:end])
		out.extend(contour)
		start = end

	return out

# Optimizer

def can_iup_in_between(deltas, coords, i, j, tolerance):
	assert j - i >= 2
	interp = list(iup_segment(coords[i+1:j], coords[i], deltas[i], coords[j], deltas[j]))
	deltas = deltas[i+1:j]

	assert len(deltas) == len(interp)

	return all(abs(complex(x-p, y-q)) <= tolerance for (x,y),(p,q) in zip(deltas, interp))

def _iup_contour_bound_forced_set(delta, coords, tolerance=0):
	"""The forced set is a conservative set of points on the contour that must be encoded
	explicitly (ie. cannot be interpolated).  Calculating this set allows for significantly
	speeding up the dynamic-programming, as well as resolve circularity in DP.

	The set is precise; that is, if an index is in the returned set, then there is no way
	that IUP can generate delta for that point, given coords and delta.
	"""
	assert len(delta) == len(coords)

	forced = set()
	# Track "last" and "next" points on the contour as we sweep.
	nd, nc = delta[0], coords[0]
	ld, lc = delta[-1], coords[-1]
	for i in range(len(delta)-1, -1, -1):
		d, c = ld, lc
		ld, lc = delta[i-1], coords[i-1]

		for j in (0,1): # For X and for Y
			cj = c[j]
			dj = d[j]
			lcj = lc[j]
			ldj = ld[j]
			ncj = nc[j]
			ndj = nd[j]

			if lcj <= ncj:
				c1, c2 = lcj, ncj
				d1, d2 = ldj, ndj
			else:
				c1, c2 = ncj, lcj
				d1, d2 = ndj, ldj

			# If coordinate for current point is between coordinate of adjacent
			# points on the two sides, but the delta for current point is NOT
			# between delta for those adjacent points (considering tolerance
			# allowance), then there is no way that current point can be IUP-ed.
			# Mark it forced.
			force = False
			if c1 <= cj <= c2:
				if not (min(d1,d2)-tolerance <= dj <= max(d1,d2)+tolerance):
					force = True
			else: # cj < c1 or c2 < cj
				if c1 == c2:
					if d1 == d2:
						if abs(dj - d1) > tolerance:
							force = True
					else:
						if abs(dj) > tolerance:
							# Disabled the following because the "d1 == d2" does
							# check does not take tolerance into consideration...
							pass # force = True
				elif d1 != d2:
					if cj < c1:
						if dj != d1 and ((dj-tolerance < d1) != (d1 < d2)):
							force = True
					else: # c2 < cj
						if d2 != dj and ((d2 < dj+tolerance) != (d1 < d2)):
							force = True

			if force:
				forced.add(i)
				break

		nd, nc = d, c

	return forced

def _iup_contour_optimize_dp(delta, coords, forced={}, tolerance=0, lookback=None):
	"""Straightforward Dynamic-Programming.  For each index i, find least-costly encoding of
	points 0 to i where i is explicitly encoded.  We find this by considering all previous
	explicit points j and check whether interpolation can fill points between j and i.

	Note that solution always encodes last point explicitly.  Higher-level is responsible
	for removing that restriction.

	As major speedup, we stop looking further whenever we see a "forced" point."""

	n = len(delta)
	if lookback is None:
		lookback = n
	costs = {-1:0}
	chain = {-1:None}
	for i in range(0, n):
		best_cost = costs[i-1] + 1

		costs[i] = best_cost
		chain[i] = i - 1

		if i - 1 in forced:
			continue

		for j in range(i-2, max(i-lookback, -2), -1):

			cost = costs[j] + 1

			if cost < best_cost and can_iup_in_between(delta, coords, j, i, tolerance):
				costs[i] = best_cost = cost
				chain[i] = j

			if j in forced:
				break

	return chain, costs

def _rot_list(l, k):
	"""Rotate list by k items forward.  Ie. item at position 0 will be
	at position k in returned list.  Negative k is allowed."""
	n = len(l)
	k %= n
	if not k: return l
	return l[n-k:] + l[:n-k]

def _rot_set(s, k, n):
	k %= n
	if not k: return s
	return {(v + k) % n for v in s}

def iup_contour_optimize(delta, coords, tolerance=0.):
	n = len(delta)

	# Get the easy cases out of the way:

	# If all are within tolerance distance of 0, encode nothing:
	if all(abs(complex(*p)) <= tolerance for p in delta):
		return [None] * n

	# If there's exactly one point, return it:
	if n == 1:
		return delta

	# If all deltas are exactly the same, return just one (the first one):
	d0 = delta[0]
	if all(d0 == d for d in delta):
		return [d0] + [None] * (n-1)

	# Else, solve the general problem using Dynamic Programming.

	forced = _iup_contour_bound_forced_set(delta, coords, tolerance)
	# The _iup_contour_optimize_dp() routine returns the optimal encoding
	# solution given the constraint that the last point is always encoded.
	# To remove this constraint, we use two different methods, depending on
	# whether forced set is non-empty or not:

	if forced:
		# Forced set is non-empty: rotate the contour start point
		# such that the last point in the list is a forced point.
		k = (n-1) - max(forced)
		assert k >= 0

		delta  = _rot_list(delta, k)
		coords = _rot_list(coords, k)
		forced = _rot_set(forced, k, n)

		chain, costs = _iup_contour_optimize_dp(delta, coords, forced, tolerance)

		# Assemble solution.
		solution = set()
		i = n - 1
		while i is not None:
			solution.add(i)
			i = chain[i]
		assert forced <= solution, (forced, solution)
		delta = [delta[i] if i in solution else None for i in range(n)]

		delta = _rot_list(delta, -k)
	else:
		# Repeat the contour an extra time, solve the 2*n case, then look for solutions of the
		# circular n-length problem in the solution for 2*n linear case.  I cannot prove that
		# this always produces the optimal solution...
		chain, costs = _iup_contour_optimize_dp(delta+delta, coords+coords, forced, tolerance, n)
		best_sol, best_cost = None, n+1

		for start in range(n-1, 2*n-1):
			# Assemble solution.
			solution = set()
			i = start
			while i > start - n:
				solution.add(i % n)
				i = chain[i]
			if i == start - n:
				cost = costs[start] - costs[start - n]
				if cost <= best_cost:
					best_sol, best_cost = solution, cost

		delta = [delta[i] if i in best_sol else None for i in range(n)]


	return delta

def iup_delta_optimize(delta, coords, ends, tolerance=0.):
	assert sorted(ends) == ends and len(coords) == (ends[-1]+1 if ends else 0) + 4
	n = len(coords)
	ends = ends + [n-4, n-3, n-2, n-1]
	out = []
	start = 0
	for end in ends:
		contour = iup_contour_optimize(delta[start:end+1], coords[start:end+1], tolerance)
		assert len(contour) == end - start + 1
		out.extend(contour)
		start = end+1

	return out
