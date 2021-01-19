package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "github.com/tensorflow/tensorflow/tensorflow/go/op"
)

var numericColumns = []string{"type", "state"}
var categoricalColumns = []string{"color"}

func main() {

	ftest, err := os.Open("test1_copy.csv")
	if err != nil {

		log.Fatal(err)
	}

	ftrain, err := os.Open("test1.csv")
	if err != nil {

		log.Fatal(err)
	}

	rtest, err := csv.NewReader(ftest).ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	rtrain, err := csv.NewReader(ftrain).ReadAll()
	log.Println(rtrain)
	if err != nil {
		log.Fatal(err)
	}
	// show()

	testY := getY(rtest)
	trainY := getY(rtrain)
	log.Println(testY, trainY)
}

func getY(train [][]string) []string {
	var trainY = []string{}
	for i := 0; i < len(train); i++ {
		thisRow := train[i]
		row, _ := thisRow[len(thisRow)-1], thisRow[:len(thisRow)-1]
		trainY = append(trainY, row)
	}
	return trainY
}

func show() {
	f, err := os.Open("test1.csv")

	if err != nil {

		log.Fatal(err)
	}

	r := csv.NewReader(f)

	record, err := r.ReadAll()
	log.Println("record: ", record)

	for value := range record {
		fmt.Printf("%s\n", record[value])
	}

}

func intputFn(features []string, labels []string, training bool, batchSize int) {
	dataset := tf.
}
