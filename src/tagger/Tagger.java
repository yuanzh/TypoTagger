package tagger;

import typo.TypoClassifier;
import utils.*;
import java.util.*;

public class Tagger {
	List<Entry>[][][] thetaToFeature;				// features appeared in each theta, [state][state], [state][word]
	HashMap<String, Integer> featureIndex;
	int featureNum;
	double[] weights;
	double[][][] weightsToTheta;					// store values of theta, [state][state], [state][word]
	double[][][] logTheta;							// store log values of theta
	List<Entry>[][] weightedSumThetaToFeature;		// store values of \sum_{d'} \theta_{c,d',t}f(c,d',t), [type][state]
	HashMap<String, Integer> wordIndex;
	int wordNum;
	String[] wordStr;

	TypoClassifier typo;
	String[] stateStr;
	int stateNum;
	int startState, endState, wordState;
	
	// data
	int[][] sentences;								// [sent][n]
	int sentNum;
	
	// for gradient
	double learnRate;
	double[] gradient;
	double[][][] standardForward;					// [sent][n][stateNum], p(z_k, y_{1:k}), log value
	double[][][] standardBackward;					// [sent][n][stateNum], p(y_{k+1:n}|z_k), log value
	double[][][] oneFixedForward;					// [sent][n][stateNum], p(z_k, y_{1:k}), log value
	double[][][] oneFixedBackward;					// [sent][n][stateNum], p(y_{k+1:n}|z_k), log value
	
	public void initialize() {
		typo = new TypoClassifier();
		typo.initialize();
		stateStr = typo.stateStr;
		stateNum = stateStr.length;
		startState = 0;
		Utils.Assert(stateStr[startState].equals("s"));
		endState = 1;
		Utils.Assert(stateStr[endState].equals("e"));
		wordState = endState + 1;
		
		// initialize vocabulary
		wordIndex = new HashMap<String, Integer>();
		wordIndex.put("a", 0);
		wordIndex.put("b", 1);
		wordIndex.put("c", 2);
		wordNum = wordIndex.size();
		wordStr = new String[wordNum];
		for (String s : wordIndex.keySet())
			wordStr[wordIndex.get(s)] = s;
		
		// initialize features
		featureIndex = new HashMap<String, Integer>();
		for (int i = 0; i < stateNum; ++i) {
			for (int j = 0; j < stateNum; ++j) {
				featureIndex.put(stateStr[i] + "," + stateStr[j], featureIndex.size());
			}
			for (int j = 0; j < wordNum; ++j) {
				featureIndex.put(stateStr[i] + "," + wordStr[j], featureIndex.size());
			}
		}
		featureNum = featureIndex.size();
		
		// initialize weight and feature arrays
		weights = new double[featureNum];
		weightsToTheta = new double[2][][];
		logTheta = new double[2][][];
		weightsToTheta[0] = new double[stateNum][stateNum];
		logTheta[0] = new double[stateNum][stateNum];
		weightsToTheta[1] = new double[stateNum][wordNum];
		logTheta[1] = new double[stateNum][stateNum];
		weightedSumThetaToFeature = new List[2][stateNum];
		for (int i = 0; i < 2; ++i)
			for (int j = 0; j < stateNum; ++j) {
				weightedSumThetaToFeature[i][j] = new ArrayList<Entry>();
			}

		thetaToFeature = new List[2][][];
		thetaToFeature[0] = new List[stateNum][stateNum];		// transition
		for (int i = 0; i < stateNum; ++i)
			for (int j = 0; j < stateNum; ++j) {
				String feature = stateStr[i] + "," + stateStr[j];
				List<Entry> list = new ArrayList<Entry>();
				list.add(new Entry(featureIndex.get(feature), 1.0));
				thetaToFeature[0][i][j] = list;
			}
		thetaToFeature[1] = new List[stateNum][wordNum];		// emission
		for (int i = 0; i < stateNum; ++i)
			for (int j = 0; j < wordNum; ++j) {
				String feature = stateStr[i] + "," + wordStr[j];
				List<Entry> list = new ArrayList<Entry>();
				list.add(new Entry(featureIndex.get(feature), 1.0));
				thetaToFeature[1][i][j] = list;
			}
		
		// initialize data
		String[] s = {"a", "c", "a", "c", "b"};
		sentences = new int[1][s.length];
		for (int i = 0; i < s.length; ++i) {
			sentences[0][i] = wordIndex.get(s[i]);
		}
		sentNum = 1;
		
		// initialize parameters
		Random r = new Random();
		for (int i = 0; i < featureNum; ++i) {
			weights[i] = r.nextDouble() * 0.02 - 0.01;
			//weights[i] = 0.0;
		}
		learnRate = 0.1;
		
		// initialize gradient arrays
		gradient = new double[featureNum];
		standardForward = new double[1][s.length][stateNum];
		standardBackward = new double[1][s.length][stateNum];
		oneFixedForward = new double[1][s.length][stateNum];
		oneFixedBackward = new double[1][s.length][stateNum];
	}
	
	public void computeTheta() {
		for (int i = 0; i < stateNum; ++i) {
			// transition
			double[] score = weightsToTheta[0][i];
			for (int j = 0; j < stateNum; ++j) {
				score[j] = 0.0;
				List<Entry> fv = thetaToFeature[0][i][j];
				for (int k = 0; k < fv.size(); ++k)
					score[j] += fv.get(k).value * weights[fv.get(k).x];
			}
			typo.softmax(score);
			for (int j = 0; j < stateNum; ++j) {
				logTheta[0][i][j] = Math.log(score[j]);
			}

			// emission
			score = weightsToTheta[1][i];
			for (int j = 0; j < wordNum; ++j) {
				score[j] = 0.0;
				List<Entry> fv = thetaToFeature[1][i][j];
				for (int k = 0; k < fv.size(); ++k)
					score[j] += fv.get(k).value * weights[fv.get(k).x];
			}
			typo.softmax(score);
			for (int j = 0; j < wordNum; ++j) {
				logTheta[1][i][j] = Math.log(score[j]);
			}
		}
	}
	
	public void computeWeightedSum() {
		for (int i = 0; i < stateNum; ++i) {
			HashMap<Integer, Entry> map = new HashMap<Integer, Entry>();
			// transition
			for (int j = 0; j < stateNum; ++j) {
				List<Entry> fv = thetaToFeature[0][i][j];
				double theta = weightsToTheta[0][i][j];
				for (int k = 0; k < fv.size(); ++k) {
					Entry fvEntry = fv.get(k);
					Entry entry = map.get(fvEntry.x);
					if (entry == null) {
						map.put(fvEntry.x, new Entry(fvEntry.x, theta * fvEntry.value));
					}
					else {
						entry.value += theta * fvEntry.value;
					}
				}
			}
			weightedSumThetaToFeature[0][i].clear();
			for (Entry entry : map.values()) {
				weightedSumThetaToFeature[0][i].add(new Entry(entry.x, entry.value));
			}
			
			// emission
			map.clear();
			for (int j = 0; j < wordNum; ++j) {
				List<Entry> fv = thetaToFeature[1][i][j];
				double theta = weightsToTheta[1][i][j];
				for (int k = 0; k < fv.size(); ++k) {
					Entry fvEntry = fv.get(k);
					Entry entry = map.get(fvEntry.x);
					if (entry == null) {
						map.put(fvEntry.x, new Entry(fvEntry.x, theta * fvEntry.value));
					}
					else {
						entry.value += theta * fvEntry.value;
					}
				}
			}
			weightedSumThetaToFeature[1][i].clear();
			for (Entry entry : map.values()) {
				weightedSumThetaToFeature[1][i].add(new Entry(entry.x, entry.value));
			}
		}
	}
	
	public void computeStandardForward(int[] sentence, double[][] forward) {
		// p(z_k, y_{1:k}), log value
		for (int i = wordState; i < stateNum; ++i) {
			forward[0][i] = logTheta[0][startState][i] + logTheta[1][i][sentence[i]];
		}
		for (int i = 1; i < sentence.length; ++i) {
			for (int j = wordState; j < stateNum; ++j) {
				forward[i][j] = Double.NEGATIVE_INFINITY;
				for (int k = wordState; k < stateNum; ++k) {
					forward[i][j] = Utils.logSumExp(forward[i][j], forward[i - 1][k] + logTheta[0][k][j] + logTheta[1][j][sentence[i]]);
				}
			}
		}
	}
	
	public void computeStandardBackward(int[] sentence, double[][] backward) {
		// p(y_{k+1:n}|z_k), log value
		int n = sentence.length;
		for (int i = wordState; i < stateNum; ++i) {
			backward[n - 1][i] = logTheta[0][i][endState];
		}
		for (int i = n - 2; i >= 0; --i) {
			for (int j = wordState; j < stateNum; ++j) {
				backward[i][j] = Double.NEGATIVE_INFINITY;
				for (int k = wordState; k < stateNum; ++k) {
					backward[i][j] = Utils.logSumExp(backward[i][j], backward[i + 1][k] + logTheta[0][j][k] + logTheta[1][k][sentence[i + 1]]);
				}
			}
		}
	}
	
	public void computeStandardTransitionEcounts(int[] sentence, double[][] forward, double[][] backward, double[][] ecounts) {
		int n = sentence.length;
		double[] score = new double[stateNum];
		// st->idx 0
		score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
		for (int d = wordState; d < stateNum; ++d) {
			score[d] = backward[0][d] + logTheta[0][startState][d] + logTheta[1][d][sentence[0]];
		}
		typo.softmax(score);
		for (int d = wordState; d < stateNum; ++d) {
			ecounts[startState][d] += score[d];
		}
		
		// idx n->en
		score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
		for (int c = wordState; c < stateNum; ++c) {
			score[c] = forward[n - 1][c] + logTheta[0][c][endState];
		}
		typo.softmax(score);
		for (int c = wordState; c < stateNum; ++c) {
			ecounts[c][endState] += score[c];
		}
		
		// idx k->idx k+1
		score = new double[stateNum * stateNum];
		for (int i = 0; i < n - 1; ++i) {
			for (int j = 0; j < stateNum; ++j) {
				score[typo.featIndex(startState, j)] = score[typo.featIndex(endState, j)] = Double.NEGATIVE_INFINITY;
				score[typo.featIndex(j, startState)] = score[typo.featIndex(j, endState)] = Double.NEGATIVE_INFINITY;
			}
			for (int c = wordState; c < stateNum; ++c) {
				for (int d = wordState; d < stateNum; ++d) {
					score[typo.featIndex(c, d)] = forward[i][c] + backward[i + 1][d] + logTheta[0][c][d] + logTheta[1][d][sentence[i + 1]];
				}
			}
			typo.softmax(score);
			for (int c = wordState; c < stateNum; ++c) {
				for (int d = wordState; d < stateNum; ++d) {
					ecounts[c][d] += score[typo.featIndex(c, d)];
				}
			}
		}
	}
	
	public void computeStandardEmissionEcounts(int[] sentence, double[][] forward, double[][] backward, double[][] ecounts) {
		int n = sentence.length;
		double[] score = new double[stateNum];
		for (int i = 0; i < n - 1; ++i) {
			score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
			for (int cp = wordState; cp < stateNum; ++cp) {
				score[cp] = forward[i][cp] + backward[i][cp];
			}
			typo.softmax(score);
			for (int cp = wordState; cp < stateNum; ++cp) {
				ecounts[cp][sentence[i]] += score[cp];
			}
		}
	}
	
	public void outputLogProbability(int[] sentence, double[][] forward, double[][] backward) {
		int n = sentence.length;
		double[] logp = new double[n];
		double[] score = new double[stateNum];
		score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
		for (int j = 0; j < n; ++j) {
			for (int k = wordState; k < stateNum; ++k) {
				score[k] = forward[j][k] + backward[j][k];
			}
			logp[j] = Double.NEGATIVE_INFINITY;
			for (int k = 0; k < stateNum; ++k) {
				logp[j] = Utils.logSumExp(logp[j], score[k]);
			}
		}
		System.out.print("Log prob: ");
		for (int j = 0; j < n; ++j) {
			System.out.printf("%.4f ", logp[j]);
		}
		System.out.println();
	}
	
	public void viterbiDecode(int[] sentence) {
		int n = sentence.length;
		double[][] opt = new double[n][stateNum];
		int[][] backtrace = new int[n][stateNum];
		for (int i = wordState; i < stateNum; ++i) {
			opt[0][i] = logTheta[0][startState][i] + logTheta[1][i][sentence[0]];
			backtrace[0][i] = -1;
		}
		for (int i = 1; i < n; ++i) {
			for (int j = wordState; j < stateNum; ++j) {
				opt[i][j] = Double.NEGATIVE_INFINITY;
				for (int k = wordState; k < stateNum; ++k) {
					double value = opt[i - 1][k] + logTheta[0][k][j] + logTheta[1][j][sentence[i]];
					if (value > opt[i][j]) {
						backtrace[i][j] = k;
						opt[i][j] = value;
					}
				}
			}
		}
		for (int i = wordState; i < stateNum; ++i) {
			opt[n - 1][i] += logTheta[0][i][endState];
		}
		
		// backtrace
		int[] pred = new int[n];
		double max = Double.NEGATIVE_INFINITY;
		for (int i = wordState; i < stateNum; ++i) {
			if (opt[n - 1][i] > max) {
				pred[n - 1] = i;
				max = opt[n - 1][i];
			}
		}
		for (int i = n - 2; i >= 0; --i) {
			pred[i] = backtrace[i + 1][pred[i + 1]];
		}
		
		// output
		for (int i = 0; i < n; ++i)
			System.out.print(stateStr[pred[i]] + "/" + wordStr[sentence[i]] + " ");
		System.out.println();
	}
	
	public void computeOneFixedForward(int[] sentence, double[][] standardForward, int c, int d, double[][] oneFixedForward) {
		// p(z_k, y_{1:k}), log value
		for (int i = wordState; i < stateNum; ++i) {
			if (c == startState && d == i)
				oneFixedForward[0][i] = logTheta[0][startState][i] + logTheta[1][i][sentence[0]];
			else
				oneFixedForward[0][i] = Double.NEGATIVE_INFINITY;
		}
		for (int i = 1; i < sentence.length; ++i) {
			for (int j = wordState; j < stateNum; ++j) {
				oneFixedForward[i][j] = Double.NEGATIVE_INFINITY;
				for (int k = wordState; k < stateNum; ++k) {
					if (k == c && j == d) {
						// current edge is the fixed one
						oneFixedForward[i][j] = Utils.logSumExp(oneFixedForward[i][j], standardForward[i - 1][k] + logTheta[0][k][j] + logTheta[1][j][sentence[i]]);
					}
					if (oneFixedForward[i - 1][k] != Double.NEGATIVE_INFINITY) {
						// current edge is not the fixed one
						oneFixedForward[i][j] = Utils.logSumExp(oneFixedForward[i][j], oneFixedForward[i - 1][k] + logTheta[0][k][j] + logTheta[1][j][sentence[i]]);
					}
				}
			}
		}
	}
	
	public void computeOneFixedBackward(int[] sentence, double[][] standardBackward, int c, int d, double[][] oneFixedBackward) {
		// p(y_{k+1:n}|z_k), log value
		int n = sentence.length;
		for (int i = wordState; i < stateNum; ++i) {
			if (c == i && d == endState)
				oneFixedBackward[n - 1][i] = logTheta[0][i][endState];
			else
				oneFixedBackward[n - 1][i] = Double.NEGATIVE_INFINITY;
		}
		for (int i = n - 2; i >= 0; --i) {
			for (int j = wordState; j < stateNum; ++j) {
				oneFixedBackward[i][j] = Double.NEGATIVE_INFINITY;
				for (int k = wordState; k < stateNum; ++k) {
					if (j == c && k == d) {
						// current edge is the fixed one
						oneFixedBackward[i][j] = Utils.logSumExp(oneFixedBackward[i][j], standardBackward[i + 1][k] + logTheta[0][j][k] + logTheta[1][k][sentence[i + 1]]);
					}
					if (oneFixedBackward[i + 1][k] != Double.NEGATIVE_INFINITY) {
						// current edge is not the fixed one
						oneFixedBackward[i][j] = Utils.logSumExp(oneFixedBackward[i][j], oneFixedBackward[i + 1][k] + logTheta[0][j][k] + logTheta[1][k][sentence[i + 1]]);
					}
				}
			}
		}
	}
	
	public void computeJointTransitionEcounts(int[] sentence, double[][] standardForward, double[][] standardBackward, 
			double[][] oneFixedForward, double[][] oneFixedBackward, int c, int d, double[][] ecounts) {
		int n = sentence.length;
		double[] score = new double[stateNum];
		// st->idx 0
		score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
		for (int dp = wordState; dp < stateNum; ++dp) {
			score[dp] = oneFixedBackward[0][dp] + logTheta[0][startState][dp] + logTheta[1][dp][sentence[0]];
		}
		typo.softmax(score);
		for (int dp = wordState; dp < stateNum; ++dp) {
			ecounts[startState][dp] += score[dp];
		}
		
		// Special case when c=c', d=d'
		if (c == startState) {
			score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
			for (int dp = wordState; dp < stateNum; ++dp) {
				score[dp] = standardBackward[0][dp] + logTheta[0][startState][dp] + logTheta[1][dp][sentence[0]];
			}
			typo.softmax(score);
			ecounts[c][d] += score[d];
		}
		
		// idx n->en
		score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
		for (int cp = wordState; cp < stateNum; ++cp) {
			score[cp] = oneFixedForward[n - 1][cp] + logTheta[0][cp][endState];
		}
		typo.softmax(score);
		for (int cp = wordState; cp < stateNum; ++cp) {
			ecounts[cp][endState] += score[cp];
		}
		
		// Special case when c=c', d=d'
		if (d == endState) {
			score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
			for (int cp = wordState; cp < stateNum; ++cp) {
				score[cp] = standardForward[n - 1][cp] + logTheta[0][cp][endState];
			}
			typo.softmax(score);
			ecounts[c][d] += score[c];
		}

		// idx k->idx k+1
		score = new double[stateNum * stateNum];
		for (int i = 0; i < n - 1; ++i) {
			for (int j = 0; j < stateNum; ++j) {
				score[typo.featIndex(startState, j)] = score[typo.featIndex(endState, j)] = Double.NEGATIVE_INFINITY;
				score[typo.featIndex(j, startState)] = score[typo.featIndex(j, endState)] = Double.NEGATIVE_INFINITY;
			}
			for (int cp = wordState; cp < stateNum; ++cp) {
				for (int dp = wordState; dp < stateNum; ++dp) {
					score[typo.featIndex(cp, dp)] = Utils.logSumExp(
							oneFixedForward[i][cp] + standardBackward[i + 1][dp] + logTheta[0][cp][dp] + logTheta[1][dp][sentence[i + 1]],
							standardForward[i][cp] + oneFixedBackward[i + 1][dp] + logTheta[0][cp][dp] + logTheta[1][dp][sentence[i + 1]]);
				}
			}
			typo.softmax(score);
			for (int cp = wordState; cp < stateNum; ++cp) {
				for (int dp = wordState; dp < stateNum; ++dp) {
					ecounts[cp][dp] += score[typo.featIndex(cp, dp)];
				}
			}
			
			// Special case when c=c', d=d'
			if (c >= wordState && d >= wordState) {
				for (int j = 0; j < stateNum; ++j) {
					score[typo.featIndex(startState, j)] = score[typo.featIndex(endState, j)] = Double.NEGATIVE_INFINITY;
					score[typo.featIndex(j, startState)] = score[typo.featIndex(j, endState)] = Double.NEGATIVE_INFINITY;
				}
				for (int cp = wordState; cp < stateNum; ++cp) {
					for (int dp = wordState; dp < stateNum; ++dp) {
						score[typo.featIndex(cp, dp)] = standardForward[i][cp] + standardBackward[i + 1][dp] + logTheta[0][cp][dp] + logTheta[1][dp][sentence[i + 1]];
					}
				}
				typo.softmax(score);
				ecounts[c][d] += score[typo.featIndex(c, d)];
			}
		}
	}
	
	public void computeJointEmissionEcounts(int[] sentence, double[][] standardForward, double[][] standardBackward, 
			double[][] oneFixedForward, double[][] oneFixedBackward, int c, int d, double[][] ecounts) {
		int n = sentence.length;
		double[] score = new double[stateNum];
		for (int i = 0; i < n - 1; ++i) {
			score[startState] = score[endState] = Double.NEGATIVE_INFINITY;
			for (int cp = wordState; cp < stateNum; ++cp) {
				score[cp] = Utils.logSumExp(
							oneFixedForward[i][cp] + standardBackward[i][cp],
							standardForward[i][cp] + oneFixedBackward[i][cp]);
			}
			typo.softmax(score);
			for (int cp = wordState; cp < stateNum; ++cp) {
				ecounts[cp][sentence[i]] += score[cp];
			}
		}
	}
	
	public void computeGradient(double[][] ecounts) {
		double[][] backGradient = typo.gradient(ecounts);		// stateNum * stateNum
//		System.out.println("back gradient");
//		for (int i = 0; i < stateNum; ++i)
//			for (int j = 0; j < stateNum; ++j)
//				System.out.printf("%d:%d:%.3f/%.3f ", i, j, backGradient[i][j], ecounts[i][j]);
//		System.out.println();
		
		double[][][] jointEcounts = new double[2][][];			// for e(c, d, c', d', t')
		jointEcounts[0] = new double[stateNum][stateNum];
		jointEcounts[1] = new double[stateNum][wordNum];
		double[][][] singleEcounts = new double[2][][];			// for e(c', d', t')
		singleEcounts[0] = new double[stateNum][stateNum];
		singleEcounts[1] = new double[stateNum][wordNum];
		double[][][] coeff = new double[2][][];					// for a(c', d', t')
		coeff[0] = new double[stateNum][stateNum];
		coeff[1] = new double[stateNum][wordNum];
		
		for (int i = 0; i < featureNum; ++i)
			gradient[i] = 0.0;
		// loop over each c, d, O(l^2)
		for (int c = 0; c < stateNum; ++c) {
			for (int d = 0; d < stateNum; ++d) {
				for (int i = 0; i < stateNum; ++i) {
					for (int j = 0; j < stateNum; ++j) {
						jointEcounts[0][i][j] = singleEcounts[0][i][j] = coeff[0][i][j] = 0.0;
					}
					for (int j = 0; j < wordNum; ++j) {
						jointEcounts[1][i][j] = singleEcounts[1][i][j] = coeff[1][i][j] = 0.0;
					}
				}

				for (int i = 0; i < sentNum; ++i) {
					// compute forward and backward that one edge is fixed to (c, d), O(nl^2)
					computeOneFixedForward(sentences[i], standardForward[i], c, d, oneFixedForward[i]);
					computeOneFixedBackward(sentences[i], standardBackward[i], c, d, oneFixedBackward[i]);
					
					// compute e(c, d, c', d', t'), O(nl^2)
					computeJointTransitionEcounts(sentences[i], standardForward[i], standardBackward[i], oneFixedForward[i], oneFixedBackward[i], c, d, jointEcounts[0]);
					computeJointEmissionEcounts(sentences[i], standardForward[i], standardBackward[i], oneFixedForward[i], oneFixedBackward[i], c, d, jointEcounts[1]);
					
					// compute e(c', d', t'),  O(nl^2)
					computeStandardTransitionEcounts(sentences[i], standardForward[i], standardBackward[i], singleEcounts[0]);
					computeStandardEmissionEcounts(sentences[i], standardForward[i], standardBackward[i], singleEcounts[1]);
				}
				
				// compute a(c', d', t') = backGradient(c, d) * (e(c, d, c', d', t') - e(c, d)e(c', d', t')), O(l^2)
				for (int cp = 0; cp < stateNum; ++cp) {
					for (int dp = 0; dp < stateNum; ++dp) {
						coeff[0][cp][dp] = backGradient[c][d] * (jointEcounts[0][cp][dp] - ecounts[c][d] * singleEcounts[0][cp][dp]);
					}
					for (int dp = 0; dp < wordNum; ++dp) {
						coeff[1][cp][dp] = backGradient[c][d] * (jointEcounts[1][cp][dp] - ecounts[c][d] * singleEcounts[1][cp][dp]);
					}
				}
				
				// add a(c', d', t') * f(c', d', t'), O(kl^2), k is the size of the feature vector
				for (int cp = 0; cp < stateNum; ++cp) {
					for (int dp = 0; dp < stateNum; ++dp) {
						List<Entry> fv = thetaToFeature[0][cp][dp];
						for (int j = 0; j < fv.size(); ++j) {
							gradient[fv.get(j).x] += coeff[0][cp][dp] * fv.get(j).value;
						}
					}
					for (int dp = 0; dp < wordNum; ++dp) {
						List<Entry> fv = thetaToFeature[1][cp][dp];
						for (int j = 0; j < fv.size(); ++j) {
							gradient[fv.get(j).x] += coeff[1][cp][dp] * fv.get(j).value;
						}
					}
				}
				
				// add -a(c', d', t') * \sum_{d''} \theta_{c', d'', t'} f(c', d'', t'), ~O(kl^3)
				for (int cp = 0; cp < stateNum; ++cp) {
					for (int dp = 0; dp < stateNum; ++dp) {
						List<Entry> fv = weightedSumThetaToFeature[0][cp];
						for (int j = 0; j < fv.size(); ++j) {
							gradient[fv.get(j).x] -= coeff[0][cp][dp] * fv.get(j).value;
						}
					}
					for (int dp = 0; dp < wordNum; ++dp) {
						List<Entry> fv = weightedSumThetaToFeature[1][cp];
						for (int j = 0; j < fv.size(); ++j) {
							gradient[fv.get(j).x] -= coeff[1][cp][dp] * fv.get(j).value;
						}
					}
				}
				
			}
		}
	}
	
	public void updateParameters() {
		for (int i = 0; i < featureNum; ++i) {
			weights[i] += learnRate * gradient[i];
		}
	}
	
	public void train() {
		int maxEpochs = 100;
		double[][] ecounts = new double[stateNum][stateNum];
		learnRate = 0.1;
		for (int epoch = 0; epoch < maxEpochs; ++epoch) {
			System.out.println("Epoch " + epoch);
			for (int i = 0; i < stateNum; ++i) {
				for (int j = 0; j < stateNum; ++j) {
					ecounts[i][j] = 0.0;
				}
			}
			
			// compute theta
			computeTheta();
			
			// compute theta weighted sum
			computeWeightedSum();
			
			// loop over each sentence
			for (int i = 0; i < sentNum; ++i) {
				// compute standard forward and backward, O(nl^2)
				computeStandardForward(sentences[i], standardForward[i]);
				computeStandardBackward(sentences[i], standardBackward[i]);
				
				// compute expected counts e(c, d) of each transition, O(nl^2)
				computeStandardTransitionEcounts(sentences[i], standardForward[i], standardBackward[i], ecounts);
				
				// compute log probability and viterbi decode
				outputLogProbability(sentences[i], standardForward[i], standardBackward[i]);
				viterbiDecode(sentences[i]);
			}
			System.out.println("obj: " + typo.entropy(typo.computeScore(ecounts)));

			// compute gradient
			computeGradient(ecounts);
			
			// update parameters
			updateParameters();
		}
		
	}
	
	public static void main(String[] args) {
		Tagger tagger = new Tagger();
		tagger.initialize();
		tagger.train();
	}

}
