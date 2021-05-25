package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

type IDPair struct {
	i, j int
}

type BitermPrediction struct {
	term1, term2               string
	termTopicID1, termTopicID2 int
	topicID1, topicID2         int
	emergingness               float64
	ascentProbablity           float64
}

type OntologyNode struct {
	topicID       int
	superTopicIDs []int
	ancestorIDs   map[int]bool
	terms         []string
}

type TermHash struct {
	terms   [][]string
	nodeIDs []int
	hash    map[string][]int
}

type LocPair struct {
	i1, j1, i2, j2 int
}

type EvaluationSummary struct {
	accuracy float64
	score    float64
	support  int
}

func fourYearSlope(hitsOfYears []int) float64 {
	return 0.3*float64(hitsOfYears[3]-hitsOfYears[0]) + 0.1*float64(hitsOfYears[2]-hitsOfYears[1])
}

func fourYearSlopeF(hitsOfYears []float64) float64 {
	return 0.3*float64(hitsOfYears[3]-hitsOfYears[0]) + 0.1*float64(hitsOfYears[2]-hitsOfYears[1])
}

func loadOntology(fileName string) []OntologyNode {
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		log.Fatalf("file %s is empty\n", fileName)
	}
	firstLine := scanner.Text()
	numNodes, err := strconv.ParseInt(firstLine, 10, 64)
	if err != nil {
		log.Fatalln(err)
	}
	ontology := []OntologyNode{}
	for idxNode := 0; idxNode < int(numNodes); idxNode++ {
		if !scanner.Scan() {
			log.Fatalf("error reading node %d of %d from file %s\n", idxNode, numNodes, fileName)
		}
		line := scanner.Text()
		fields := strings.Split(line, ",")
		id, err := strconv.ParseInt(fields[0], 10, 64)
		if err != nil {
			log.Fatalln(err)
		}
		if int(id) != idxNode {
			log.Fatalf("expected node id %d, got %d at line %d of file %s\n", idxNode, id, id+1, fileName)
		}
		terms := fields[1:]
		ontology = append(ontology, OntologyNode{int(id), []int{}, map[int]bool{}, terms})
	}
	for numNodesLinked := 0; numNodesLinked < int(numNodes); numNodesLinked++ {
		if !scanner.Scan() {
			log.Fatalf("error reading node-links %d of %d from file %s\n", numNodesLinked, numNodes, fileName)
		}
		line := scanner.Text()
		fields := strings.Split(line, ",")
		id, err := strconv.ParseInt(fields[0], 10, 64)
		if err != nil {
			log.Fatalln(err)
		}
		superTopicIDs := []int{}
		numFields := len(fields)
		for i := 1; i < numFields; i++ {
			superTopicID, err := strconv.ParseInt(fields[i], 10, 64)
			if err != nil {
				log.Fatalln(err)
			}
			superTopicIDs = append(superTopicIDs, int(superTopicID))
		}
		ontology[id].superTopicIDs = superTopicIDs
	}
	for nodeID := 0; nodeID < int(numNodes); nodeID++ {
		boundary := map[int]bool{}
		for _, superTopicID := range ontology[nodeID].superTopicIDs {
			boundary[superTopicID] = true
		}
		ancestors := map[int]bool{}
		for len(boundary) > 0 {
			newBoundary := map[int]bool{}
			for ancestorID := range boundary {
				ancestors[ancestorID] = true
				for _, grandAncestorID := range ontology[ancestorID].superTopicIDs {
					newBoundary[grandAncestorID] = true
				}
			}
			boundary = newBoundary
		}
		ontology[nodeID].ancestorIDs = ancestors
	}
	return ontology
}

func createTermHash(ontology []OntologyNode) TermHash {
	terms := [][]string{}
	nodeIDs := []int{}
	hash := make(map[string][]int)
	for nodeID, node := range ontology {
		for _, term := range node.terms {
			words := strings.Split(term, " ")
			termID := len(terms)
			terms = append(terms, words)
			nodeIDs = append(nodeIDs, nodeID)
			hashedTermIDs, exists := hash[words[0]]
			if !exists {
				hashedTermIDs = []int{}
			}
			hashedTermIDs = append(hashedTermIDs, termID)
			hash[words[0]] = hashedTermIDs
		}
	}
	return TermHash{terms, nodeIDs, hash}
}

func loadNormalizedWordVectors(fileName string, termHash TermHash) map[int][]float64 {
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		log.Fatalf("file %s is empty\n", fileName)
	}
	firstLine := scanner.Text()
	fieldsOfFirstLine := strings.Split(firstLine, " ")
	if len(fieldsOfFirstLine) != 2 {
		log.Fatalf("format of file %s is incorrect\n", fileName)
	}
	vocabSize, err := strconv.ParseInt(fieldsOfFirstLine[0], 10, 64)
	if err != nil {
		log.Fatalln(err)
	}
	numDims, err := strconv.ParseInt(fieldsOfFirstLine[1], 10, 64)
	if err != nil {
		log.Fatalln(err)
	}
	vectors := map[int][]float64{}
	for i := 0; i < int(vocabSize); i++ {
		if !scanner.Scan() {
			log.Fatalf("error reading vector %d from file %s\n", i, fileName)
		}
		line := scanner.Text()
		fields := strings.Split(line, " ")
		if len(fields) < int(numDims)+1 {
			log.Println(line)
			log.Fatalf("error reading vector %d from file %s\n", i, fileName)
		}
		vector := make([]float64, numDims)
		norm := 0.0
		for j := 0; j < int(numDims); j++ {
			component, err := strconv.ParseFloat(fields[j+1], 64)
			if err != nil {
				log.Fatalln(err)
			}
			vector[j] = component
			norm += component * component
		}
		norm = math.Sqrt(norm)
		if norm > 0.0 {
			for j := 0; j < int(numDims); j++ {
				vector[j] /= norm
			}
		}

		words1 := strings.Split(fields[0], "_")
		words := []string{}
		for _, word := range words1 {
			subwords := strings.Split(word, "-")
			for _, subword := range subwords {
				if subword != "" {
					words = append(words, subword)
				}
			}
		}
		if len(words) == 0 {
			continue
		}
		if words[0] == "" {
			continue
		}
		ids, exists := termHash.hash[words[0]]
		if exists {
			for _, id := range ids {
				words2 := termHash.terms[id]
				if len(words) != len(words2) {
					continue
				}
				matched := true
				for j := 0; j < len(words); j++ {
					if words[j] != words2[j] {
						matched = false
						break
					}
				}
				if matched {
					vectors[id] = vector
				}
			}
		}
	}

	return vectors
}

func hyphenedTerm(termHash TermHash, i int) string {
	result := termHash.terms[i][0]
	for j := 1; j < len(termHash.terms[i]); j++ {
		result += "-" + termHash.terms[i][j]
	}
	return result
}

func leap2trend(name string, termHash TermHash) []BitermPrediction {
	vectorsOfYears := make([]map[int][]float64, 4)
	for year := 2015; year <= 2018; year++ {
		fileName := fmt.Sprintf("input/word_embedding/%s-%d.txt", name, year)
		vectorsOfYears[year-2015] = loadNormalizedWordVectors(fileName, termHash)
	}
	idOfRows := []int{}
	for id := range vectorsOfYears[3] {
		idOfRows = append(idOfRows, id)
	}
	numIDs := len(idOfRows)
	numIDPairs := numIDs * (numIDs - 1) / 2
	numDims := len(vectorsOfYears[3][idOfRows[0]])
	defaultVec := make([]float64, numDims)
	for j := 0; j < numDims; j++ {
		defaultVec[j] = 0.0
	}

	rankMatOfYears := make([][]int, 4)
	idPairs0 := []IDPair{}
	for i := 0; i < numIDs-1; i++ {
		for j := i + 1; j < numIDs; j++ {
			idPairs0 = append(idPairs0, IDPair{i, j})
		}
	}
	if len(idPairs0) != numIDPairs {
		log.Fatalln("incorrect numIDPairs")
	}
	for year := 2015; year <= 2018; year++ {
		simMat := make([]float64, numIDs*numIDs)
		for i, idi := range idOfRows {
			vi, exists := vectorsOfYears[year-2015][idi]
			if !exists {
				vi = defaultVec
			}
			for j, idj := range idOfRows {
				if i == j {
					simMat[i*numIDs+j] = 1.0
					continue
				}

				vj, exists := vectorsOfYears[year-2015][idj]
				if !exists {
					vj = defaultVec
				}
				sim := 0.0
				for k := 0; k < numDims; k++ {
					sim += vi[k] * vj[k]
				}

				simMat[i*numIDs+j] = sim
			}
		}

		idPairs := make([]IDPair, numIDPairs)
		for pairID := 0; pairID < numIDPairs; pairID++ {
			idPairs[pairID] = idPairs0[pairID]
		}

		sort.Slice(idPairs, func(x, y int) bool {
			// must use "<" to ensure larger rank for higher similarity
			return simMat[idPairs[x].i*numIDs+idPairs[x].j] < simMat[idPairs[y].i*numIDs+idPairs[y].j]
		})

		rankMat := make([]int, numIDs*numIDs)
		for rank, idPair := range idPairs {
			rankMat[idPair.i*numIDs+idPair.j] = rank
		}
		rankMatOfYears[year-2015] = rankMat
	}

	slopes := make([]float64, numIDPairs)
	ranks := make([]int, numIDPairs)
	for pairID := 0; pairID < numIDPairs; pairID++ {
		offset := idPairs0[pairID].i*numIDs + idPairs0[pairID].j
		hits := []int{
			rankMatOfYears[0][offset],
			rankMatOfYears[1][offset],
			rankMatOfYears[2][offset],
			rankMatOfYears[3][offset],
		}
		slopes[pairID] = fourYearSlope(hits)
		ranks[pairID] = pairID
	}
	sort.Slice(ranks, func(x, y int) bool {
		return slopes[ranks[x]] > slopes[ranks[y]]
	})

	result := make([]BitermPrediction, numIDPairs)
	for k := 0; k < numIDPairs; k++ {
		pairK := idPairs0[ranks[k]]
		slope := slopes[ranks[k]]
		ascentProbablity := 1.0
		if slope <= 0.0 {
			ascentProbablity = 0.0
		}
		topicIDi := termHash.nodeIDs[pairK.i]
		topicIDj := termHash.nodeIDs[pairK.j]
		if topicIDi < topicIDj {
			result[k] = BitermPrediction{
				hyphenedTerm(termHash, pairK.i),
				hyphenedTerm(termHash, pairK.j),
				topicIDi,
				topicIDj,
				topicIDi,
				topicIDj,
				slope,
				ascentProbablity,
			}
		} else {
			result[k] = BitermPrediction{
				hyphenedTerm(termHash, pairK.i),
				hyphenedTerm(termHash, pairK.j),
				topicIDi,
				topicIDj,
				topicIDj,
				topicIDi,
				slope,
				ascentProbablity,
			}
		}

	}
	return result
}

func saveBitermPrediction(fileName string, bitermPredict []BitermPrediction) {
	file, err := os.Create(fileName)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()
	numToSave := len(bitermPredict)
	for bitermID := 0; bitermID < numToSave; bitermID++ {
		term1 := bitermPredict[bitermID].term1
		term2 := bitermPredict[bitermID].term2
		topic1 := bitermPredict[bitermID].topicID1
		topic2 := bitermPredict[bitermID].topicID2
		emergingness := bitermPredict[bitermID].emergingness
		prob := bitermPredict[bitermID].ascentProbablity
		file.WriteString(fmt.Sprintf("%s %s %d %d %f %f\n", term1, term2, topic1, topic2, emergingness, prob))
	}
}

func findTermMatches(words []string, termHash TermHash) map[IDPair]LocPair {
	hits := map[int]int{}
	n := len(words)
	for i, w := range words {
		termIDs, exists := termHash.hash[w]
		if !exists {
			continue
		}
		for _, termID := range termIDs {
			termWords := termHash.terms[termID]
			m := len(termWords)
			if i+m > n {
				continue
			}
			matched := true
			for j := 1; j < m; j++ {
				if termWords[j] != words[i+j] {
					matched = false
					break
				}
			}
			if matched {
				hits[termID] = i
			}
		}
	}

	pairMatches := map[IDPair]LocPair{}
	for termID1, i1 := range hits {
		j1 := i1 + len(termHash.terms[termID1])
		for termID2, i2 := range hits {
			if termID1 < termID2 {
				j2 := i2 + len(termHash.terms[termID2])
				if j1 <= i2 || j2 <= i1 {
					pairMatches[IDPair{termID1, termID2}] = LocPair{
						i1: i1,
						j1: j1,
						i2: i2,
						j2: j2,
					}
				}
			}
		}
	}

	return pairMatches
}

func reverseLoc(locPair LocPair) LocPair {
	return LocPair{
		i1: locPair.i2,
		j1: locPair.j2,
		i2: locPair.i1,
		j2: locPair.j1,
	}
}

func findTopicMatches(termMatches map[IDPair]LocPair, termHash TermHash) map[IDPair]LocPair {
	topicMatches := map[IDPair]LocPair{}
	for termIDPair, locPair := range termMatches {
		nodeID1 := termHash.nodeIDs[termIDPair.i]
		nodeID2 := termHash.nodeIDs[termIDPair.j]
		if nodeID1 != nodeID2 {
			if nodeID1 < nodeID2 {
				topicMatches[IDPair{nodeID1, nodeID2}] = locPair
			} else {
				topicMatches[IDPair{nodeID2, nodeID1}] = reverseLoc(locPair)
			}
		}
	}
	return topicMatches
}

func getTopicIDs(nodeID int, ontology []OntologyNode) map[int]bool {
	result := map[int]bool{nodeID: true}
	for ancestorID := range ontology[nodeID].ancestorIDs {
		result[ancestorID] = true
	}
	return result
}

func findMultiResTopicMatches(termMatches map[IDPair]LocPair, termHash TermHash, ontology []OntologyNode) map[IDPair]LocPair {
	topicMatches := map[IDPair]LocPair{}
	for termIDPair, locPair := range termMatches {
		nodeID1 := termHash.nodeIDs[termIDPair.i]
		nodeID2 := termHash.nodeIDs[termIDPair.j]
		topicIDs1 := getTopicIDs(nodeID1, ontology)
		topicIDs2 := getTopicIDs(nodeID2, ontology)
		for topicID1 := range topicIDs1 {
			for topicID2 := range topicIDs2 {
				if topicID1 == topicID2 {
					continue
				}
				_, isAncestor1 := ontology[topicID1].ancestorIDs[topicID2]
				_, isAncestor2 := ontology[topicID2].ancestorIDs[topicID1]
				if !isAncestor1 && !isAncestor2 {
					if topicID1 < topicID2 {
						topicMatches[IDPair{topicID1, topicID2}] = locPair
					} else {
						topicMatches[IDPair{topicID2, topicID1}] = reverseLoc(locPair)
					}
				}
			}
		}
	}

	return topicMatches
}

func findPairMatches(fileName string, multiRes bool, termHash TermHash, ontology []OntologyNode) (map[IDPair]map[int]LocPair, []int, []string) {
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	pairMatches := map[IDPair]map[int]LocPair{}
	years := []int{}
	titles := []string{}
	scanner := bufio.NewScanner(file)
	titleID := 0
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, ". ")
		if len(fields) > 1 {
			if len(fields) != 2 {
				log.Fatalf("too many fields in %s\n", line)
			}
			year, err := strconv.Atoi(fields[0])
			if err != nil {
				log.Fatalln(err)
			}
			title := fields[1]
			years = append(years, year)
			titles = append(titles, title)

			words := strings.Split(title, " ")
			termMatches := findTermMatches(words, termHash)
			topicMatches := func() map[IDPair]LocPair {
				if multiRes {
					return findMultiResTopicMatches(termMatches, termHash, ontology)
				} else {
					return findTopicMatches(termMatches, termHash)
				}
			}()
			for idPair, locPair := range topicMatches {
				hits, exists := pairMatches[idPair]
				if !exists {
					hits = map[int]LocPair{}
				}
				hits[titleID] = locPair

				pairMatches[idPair] = hits
			}
			titleID++
		}
	}
	return pairMatches, years, titles
}

func countPairHits(pairMatches map[IDPair]map[int]LocPair, years []int) map[IDPair]map[int]int {
	pairHits := map[IDPair]map[int]int{}
	for idPair, titleHits := range pairMatches {
		yearHits := map[int]int{}
		for titleID := range titleHits {
			year := years[titleID]
			oldHit, exists := yearHits[year]
			if !exists {
				oldHit = 0
			}
			yearHits[year] = oldHit + 1
		}
		pairHits[idPair] = yearHits
	}
	return pairHits
}

func hitsOfYear(hitsOfYears map[int]int, year int) int {
	hits, exists := hitsOfYears[year]
	if !exists {
		hits = 0
	}
	return hits
}

func computeEmergingness(hitsOfYears map[int]int, toYear int) (float64, int) {
	// use sum of hitsOfYears to compute scarceness
	sumHits := 0.0
	totalSumHits := 0
	for year, hits := range hitsOfYears {
		if year <= toYear {
			sumHits += float64(hits)
		}
		totalSumHits += hits
	}
	scarceness := 1.0 / math.Sqrt(sumHits+1.0)

	// slope of 4 years to compute how much the topic is becoming frequent
	hitsOf4Years := []int{
		hitsOfYear(hitsOfYears, toYear-1),
		hitsOfYear(hitsOfYears, toYear),
		hitsOfYear(hitsOfYears, toYear+1),
		hitsOfYear(hitsOfYears, toYear+2),
	}
	futureFrequentness := fourYearSlope(hitsOf4Years)

	// definition: emerging = scarce + becoming frequent
	emergingness := scarceness * futureFrequentness
	return emergingness, totalSumHits
}

func predictEmergingness(hitsOfYears map[int]int, toYear int) (float64, float64) {
	hitsOfRecentYears := []int{
		hitsOfYear(hitsOfYears, toYear-6),
		hitsOfYear(hitsOfYears, toYear-5),
		hitsOfYear(hitsOfYears, toYear-4),
		hitsOfYear(hitsOfYears, toYear-3),
		hitsOfYear(hitsOfYears, toYear-2),
		hitsOfYear(hitsOfYears, toYear-1),
		hitsOfYear(hitsOfYears, toYear),
	}
	slopesOfRecentYears := []float64{
		fourYearSlope(hitsOfRecentYears[0:4]),
		fourYearSlope(hitsOfRecentYears[1:5]),
		fourYearSlope(hitsOfRecentYears[2:6]),
		fourYearSlope(hitsOfRecentYears[3:7]),
	}
	sslope := fourYearSlopeF(slopesOfRecentYears)

	// compute probability of ascent
	sigma := 0.0
	for i := 1; i < 3; i++ {
		diff := slopesOfRecentYears[i] - slopesOfRecentYears[i-1] - sslope
		sigma += diff * diff
	}
	sigma = math.Sqrt(sigma/3.0) + 1e-10

	mu := slopesOfRecentYears[3] + 2.0*sslope
	ascentProbablity := func() float64 {
		if 0.0 >= mu {
			return 0.5 - 0.5*math.Erf((-mu)/(2.0*sigma*math.Sqrt2))
		} else if hitsOfRecentYears[5] == 0 && hitsOfRecentYears[6] == 0 {
			return 0.0
		} else if hitsOfRecentYears[5] == 0 || hitsOfRecentYears[6] == 0 {
			return 0.25 + 0.25*math.Erf((mu)/(2.0*sigma*math.Sqrt2))
		} else {
			return 0.5 + 0.5*math.Erf((mu)/(2.0*sigma*math.Sqrt2))
		}
	}()

	futureFrequentness := mu

	// use sum of hitsOfYears to compute scarceness
	sumHits := 0.0
	for year, hits := range hitsOfYears {
		if year <= toYear {
			sumHits += float64(hits)
		}
	}
	scarceness := 1.0 / math.Sqrt(sumHits+1.0)

	// definition: emerging = scarce + becoming frequent
	emergingness := scarceness * futureFrequentness

	return emergingness, ascentProbablity
}

func hyphenedTopic(ontology []OntologyNode, topicID int) string {
	return strings.Replace(ontology[topicID].terms[0], " ", "-", -1)
}

func rankByEmergingness(pairHits map[IDPair]map[int]int, ontology []OntologyNode) []BitermPrediction {
	result := []BitermPrediction{}
	for pair, hitsOfYears := range pairHits {
		emergingness, ascentProbability := predictEmergingness(hitsOfYears, 2018)
		result = append(result, BitermPrediction{
			hyphenedTopic(ontology, pair.i),
			hyphenedTopic(ontology, pair.j),
			pair.i,
			pair.j,
			pair.i,
			pair.j,
			emergingness,
			ascentProbability,
		})
	}
	sort.Slice(result, func(x, y int) bool {
		return result[x].emergingness > result[y].emergingness
	})
	return result
}

func evaluate(fileName string, toYear int, bitermPredict []BitermPrediction, pairMatches map[IDPair]map[int]LocPair, years []int, titles []string) {
	fromYear := toYear - 7
	lastYear := toYear + 2

	// metrics to evaluate: accuracy, recall and support
	numPairs := len(bitermPredict)
	summaries := make([]EvaluationSummary, numPairs)
	pairHits := countPairHits(pairMatches, years)
	for idx, predict := range bitermPredict {
		pair := IDPair{predict.topicID1, predict.topicID2}
		if pair.i > pair.j {
			log.Fatalf("incorrect topic IDs in pair %v\n", predict)
		}
		hitsOfYears, exists := pairHits[pair]
		if !exists {
			score := 0.0
			accuracy := 1.0 - predict.ascentProbablity
			summaries[idx] = EvaluationSummary{accuracy: accuracy, score: score, support: 0}
		} else {
			score, support := computeEmergingness(hitsOfYears, 2018)
			accuracy := func() float64 {
				if score > 0.0 {
					return predict.ascentProbablity
				} else {
					return 1.0 - predict.ascentProbablity
				}
			}()
			summaries[idx] = EvaluationSummary{accuracy: accuracy, score: score, support: support}
		}
	}

	topKAccuracies := make([]float64, 30)
	topKPrecision := make([]float64, 30)
	topKRecall := make([]float64, 30)
	topKSupport := make([]float64, 30)

	for k := 100; k <= 3000; k += 100 {
		sumAccuracy := 0.0
		sumPrecision := 0.0
		sumRecall := 0.0
		sumSupport := 0.0
		n := k
		if n > numPairs {
			n = numPairs
		}
		numRealAscents := 0.0
		numPredAscents := 0.0
		for idx := 0; idx < n; idx++ {
			sumAccuracy += summaries[idx].accuracy
			numPredAscents += bitermPredict[idx].ascentProbablity
			sumPrecision += bitermPredict[idx].ascentProbablity * summaries[idx].accuracy
			if summaries[idx].score > 0.0 {
				sumRecall += summaries[idx].accuracy
				numRealAscents += 1.0
			}
			sumSupport += float64(summaries[idx].support)
		}
		topKAccuracies[k/100-1] = sumAccuracy / float64(n)
		topKPrecision[k/100-1] = (sumPrecision + 1e-10) / (numPredAscents + 1e-10)
		topKRecall[k/100-1] = (sumRecall + 1e-10) / (numRealAscents + 1e-10)
		topKSupport[k/100-1] = sumSupport / float64(n)
	}

	kLine := "k"
	accuracyLine := "accuracy"
	precicionLine := "precision"
	recallLine := "recall"
	f1scoreLine := "f1score"
	supportLine := "support"
	for kid := 0; kid < 30; kid++ {
		k := (kid + 1) * 100
		kLine += fmt.Sprintf(",%d", k)
		accuracyLine += fmt.Sprintf(",%f", topKAccuracies[kid])
		precicionLine += fmt.Sprintf(",%f", topKPrecision[kid])
		recallLine += fmt.Sprintf(",%f", topKRecall[kid])
		f1scoreLine += fmt.Sprintf(",%f", 2.0/(1.0/(topKPrecision[kid]+1e-10)+1.0/(topKRecall[kid]+1e-10)))
		supportLine += fmt.Sprintf(",%f", topKSupport[kid])
	}
	kLine += "\n"
	accuracyLine += "\n"
	precicionLine += "\n"
	recallLine += "\n"
	f1scoreLine += "\n"
	supportLine += "\n"

	file, err := os.Create(fileName)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	file.WriteString(kLine)
	file.WriteString(accuracyLine)
	file.WriteString(precicionLine)
	file.WriteString(recallLine)
	file.WriteString(f1scoreLine)
	file.WriteString(supportLine)

	for idx := 0; idx < 3000 && idx < numPairs; idx++ {
		predict := bitermPredict[idx]
		summary := summaries[idx]
		file.WriteString("\n")
		file.WriteString(fmt.Sprintf("Super-Topics: %s(%d) %s(%d)\n", predict.term1, predict.termTopicID1, predict.term2, predict.termTopicID2))
		file.WriteString(fmt.Sprintf("PredEmerging %f RealEmerging %f PredAscProb %f PredAcc %f\n", predict.emergingness, summary.score, predict.ascentProbablity, summary.accuracy))

		titleIDs := []int{}
		titleYears := []int{}
		titleRanks := []int{}
		locPairs := []LocPair{}
		hitsOfYears := make([]int, 10)
		for i := 0; i < 10; i++ {
			hitsOfYears[i] = 0
		}
		for titleID, locPair := range pairMatches[IDPair{predict.topicID1, predict.topicID2}] {
			year := years[titleID]
			titleIDs = append(titleIDs, titleID)
			titleYears = append(titleYears, year)
			titleRanks = append(titleRanks, len(titleIDs)-1)
			locPairs = append(locPairs, locPair)

			if year >= fromYear && year <= lastYear {
				hitsOfYears[year-fromYear]++
			}
		}
		sort.Slice(titleRanks, func(x, y int) bool {
			return titleYears[titleRanks[x]] > titleYears[titleRanks[y]]
		})

		yearDistributionLine := fmt.Sprintf("hitsOfYear(%d-%d)", fromYear, lastYear)
		for i := 0; i < 10; i++ {
			yearDistributionLine += fmt.Sprintf(" %d", hitsOfYears[i])
		}
		yearDistributionLine += "\n"
		file.WriteString(yearDistributionLine)

		file.WriteString("Supporting Papers:\n")
		for _, i := range titleRanks {
			titleID := titleIDs[i]
			year := years[titleID]
			title := titles[titleID]
			loc := locPairs[i]
			paperLine := fmt.Sprintf("%d.", year)
			words := strings.Split(title, " ")
			for j := 0; j < loc.i1 && j < loc.i2; j++ {
				paperLine += " " + words[j]
			}
			if loc.i1 < loc.i2 {
				paperLine += fmt.Sprintf(" <%d>", predict.topicID1)
				for j := loc.i1; j < loc.j1; j++ {
					paperLine += "-" + words[j]
				}
				for j := loc.j1; j < loc.i2; j++ {
					paperLine += " " + words[j]
				}
				paperLine += fmt.Sprintf(" <%d>", predict.topicID2)
				for j := loc.i2; j < loc.j2; j++ {
					paperLine += "-" + words[j]
				}
				for j := loc.j2; j < len(words); j++ {
					paperLine += "-" + words[j]
				}
			} else {
				paperLine += fmt.Sprintf("<%d>", predict.topicID2)
				for j := loc.i2; j < loc.j2; j++ {
					paperLine += "-" + words[j]
				}
				for j := loc.j2; j < loc.i1; j++ {
					paperLine += " " + words[j]
				}
				paperLine += fmt.Sprintf("<%d>", predict.topicID1)
				for j := loc.i1; j < loc.j1; j++ {
					paperLine += "-" + words[j]
				}
				for j := loc.j1; j < len(words); j++ {
					paperLine += "-" + words[j]
				}
			}
			paperLine += "\n"
			file.WriteString(paperLine)
		}
	}
}

func main() {
	datasetNames := []string{
		"AAAI",
		"CVPR",
		"ICML",
		"IJCAI",
		"NIPS",
	}
	ontology := loadOntology("input/ontology/cso.txt")
	termHash := createTermHash(ontology)
	for _, name := range datasetNames {
		fmt.Printf("compute leap2trend %s\n", name)
		fileNameIn := fmt.Sprintf("input/year_and_titles/%s.txt", name)
		fileNameOut := fmt.Sprintf("output/leap2trend/%s.txt", name)
		fileNameEval := fmt.Sprintf("output/comparisons/leap2trend-%s.txt", name)
		bitermPredict := leap2trend(name, termHash)
		saveBitermPrediction(fileNameOut, bitermPredict)
		fmt.Printf("evaluate leap2trend %s\n", name)
		pairMatches, years, titles := findPairMatches(fileNameIn, false, termHash, ontology)
		evaluate(fileNameEval, 2018, bitermPredict, pairMatches, years, titles)
	}
	for _, name := range datasetNames {
		fmt.Printf("compute dual-aspect %s\n", name)
		fileNameIn := fmt.Sprintf("input/year_and_titles/%s.txt", name)
		fileNameOut := fmt.Sprintf("output/dual_aspect/%s.txt", name)
		fileNameEval := fmt.Sprintf("output/comparisons/dual-aspect-%s.txt", name)
		pairMatches, years, titles := findPairMatches(fileNameIn, true, termHash, ontology)
		pairHits := countPairHits(pairMatches, years)
		bitermPredict := rankByEmergingness(pairHits, ontology)
		saveBitermPrediction(fileNameOut, bitermPredict)
		fmt.Printf("evaluate dual-aspect %s\n", name)
		evaluate(fileNameEval, 2018, bitermPredict, pairMatches, years, titles)
	}
}
