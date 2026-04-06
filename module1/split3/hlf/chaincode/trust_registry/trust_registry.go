package main

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type TrustRegistry struct{ contractapi.Contract }

type ModelRecord struct {
	Round          int     `json:"round"`
	ModelHash      string  `json:"modelHash"`
	BlockHash      string  `json:"blockHash"`
	PrevBlockHash  string  `json:"prevBlockHash"`
	GlobalF1       float64 `json:"globalF1"`
	GlobalAUC      float64 `json:"globalAUC"`
	TrustedClients []int   `json:"trustedClients"`
	FlaggedClients []int   `json:"flaggedClients"`
	ParamCount     int     `json:"paramCount"`
	TotalBytes     int     `json:"totalBytes"`
	Timestamp      int64   `json:"timestamp"`
	TxID           string  `json:"txId"`
}
type TamperAlert struct {
	ID                          int
	Round                       int
	AlertType, Detail, Severity string
	Timestamp                   int64
	TxID                        string
}
type AuditEvent struct {
	ID          int
	EventType   string
	Round       int
	Actor, Data string
	Timestamp   int64
	TxID        string
}
type VerifyResult struct {
	Verified                bool
	Round                   int
	StoredHash, ClaimedHash string
}

func (t *TrustRegistry) nextID(ctx contractapi.TransactionContextInterface, ns string) (int, error) {
	key := "CTR:" + ns
	b, err := ctx.GetStub().GetState(key)
	if err != nil {
		return 0, err
	}
	n := 0
	if b != nil {
		n, _ = strconv.Atoi(string(b))
	}
	n++
	return n, ctx.GetStub().PutState(key, []byte(strconv.Itoa(n)))
}

func (t *TrustRegistry) RegisterModel(ctx contractapi.TransactionContextInterface,
	roundStr, modelHash, blockHash, prevBlockHash, f1Str, aucStr,
	trustedJSON, flaggedJSON, paramStr, bytesStr string) error {

	round, _ := strconv.Atoi(roundStr)
	key := "MODEL:" + roundStr
	if ex, _ := ctx.GetStub().GetState(key); ex != nil {
		return fmt.Errorf("round %d already registered", round)
	}
	if round == 1 {
		if prevBlockHash != "GENESIS" {
			return fmt.Errorf("round 1 must use GENESIS")
		}
	} else {
		pb, _ := ctx.GetStub().GetState(fmt.Sprintf("MODEL:%d", round-1))
		if pb == nil {
			return fmt.Errorf("round %d not found — register sequentially", round-1)
		}
		var prev ModelRecord
		json.Unmarshal(pb, &prev)
		if prev.BlockHash != prevBlockHash {
			return fmt.Errorf("CHAIN_VIOLATION round %d", round)
		}
	}
	f1, _ := strconv.ParseFloat(f1Str, 64)
	auc, _ := strconv.ParseFloat(aucStr, 64)
	pc, _ := strconv.Atoi(paramStr)
	tb, _ := strconv.Atoi(bytesStr)
	var trusted, flagged []int
	json.Unmarshal([]byte(trustedJSON), &trusted)
	json.Unmarshal([]byte(flaggedJSON), &flagged)
	rec := ModelRecord{Round: round, ModelHash: modelHash, BlockHash: blockHash,
		PrevBlockHash: prevBlockHash, GlobalF1: f1, GlobalAUC: auc,
		TrustedClients: trusted, FlaggedClients: flagged, ParamCount: pc, TotalBytes: tb,
		Timestamp: time.Now().UnixMilli(), TxID: ctx.GetStub().GetTxID()}
	b, _ := json.Marshal(rec)
	return ctx.GetStub().PutState(key, b)
}

func (t *TrustRegistry) GetModel(ctx contractapi.TransactionContextInterface, roundStr string) (*ModelRecord, error) {
	b, err := ctx.GetStub().GetState("MODEL:" + roundStr)
	if err != nil || b == nil {
		return nil, fmt.Errorf("round %s not found", roundStr)
	}
	var rec ModelRecord
	json.Unmarshal(b, &rec)
	return &rec, nil
}

func (t *TrustRegistry) VerifyModelHash(ctx contractapi.TransactionContextInterface, roundStr, claimed string) (*VerifyResult, error) {
	rec, err := t.GetModel(ctx, roundStr)
	if err != nil {
		return &VerifyResult{Verified: false}, nil
	}
	round, _ := strconv.Atoi(roundStr)
	return &VerifyResult{Verified: rec.ModelHash == claimed, Round: round, StoredHash: rec.ModelHash, ClaimedHash: claimed}, nil
}

func (t *TrustRegistry) QueryAllModels(ctx contractapi.TransactionContextInterface) ([]*ModelRecord, error) {
	iter, err := ctx.GetStub().GetStateByRange("MODEL:", "MODEL:~")
	if err != nil {
		return nil, err
	}
	defer iter.Close()
	var out []*ModelRecord
	for iter.HasNext() {
		kv, _ := iter.Next()
		var rec ModelRecord
		if json.Unmarshal(kv.Value, &rec) == nil {
			out = append(out, &rec)
		}
	}
	return out, nil
}

func (t *TrustRegistry) RaiseTamperAlert(ctx contractapi.TransactionContextInterface, roundStr, alertType, detail, severity string) error {
	id, err := t.nextID(ctx, "ALERT")
	if err != nil {
		return err
	}
	round, _ := strconv.Atoi(roundStr)
	alert := TamperAlert{ID: id, Round: round, AlertType: alertType, Detail: detail, Severity: severity, Timestamp: time.Now().UnixMilli(), TxID: ctx.GetStub().GetTxID()}
	b, _ := json.Marshal(alert)
	return ctx.GetStub().PutState(fmt.Sprintf("ALERT:%08d", id), b)
}

func (t *TrustRegistry) GetAlerts(ctx contractapi.TransactionContextInterface) ([]*TamperAlert, error) {
	iter, err := ctx.GetStub().GetStateByRange("ALERT:", "ALERT:~")
	if err != nil {
		return nil, err
	}
	defer iter.Close()
	var out []*TamperAlert
	for iter.HasNext() {
		kv, _ := iter.Next()
		var a TamperAlert
		if json.Unmarshal(kv.Value, &a) == nil {
			out = append(out, &a)
		}
	}
	return out, nil
}

func (t *TrustRegistry) AppendEvent(ctx contractapi.TransactionContextInterface, eventType, roundStr, actor, dataJSON string) error {
	id, err := t.nextID(ctx, "AUDIT")
	if err != nil {
		return err
	}
	round, _ := strconv.Atoi(roundStr)
	ev := AuditEvent{ID: id, EventType: eventType, Round: round, Actor: actor, Data: dataJSON, Timestamp: time.Now().UnixMilli(), TxID: ctx.GetStub().GetTxID()}
	b, _ := json.Marshal(ev)
	return ctx.GetStub().PutState(fmt.Sprintf("AUDIT:%08d", id), b)
}

func (t *TrustRegistry) ExportAuditTrail(ctx contractapi.TransactionContextInterface) ([]*AuditEvent, error) {
	iter, err := ctx.GetStub().GetStateByRange("AUDIT:", "AUDIT:~")
	if err != nil {
		return nil, err
	}
	defer iter.Close()
	var out []*AuditEvent
	for iter.HasNext() {
		kv, _ := iter.Next()
		var ev AuditEvent
		if json.Unmarshal(kv.Value, &ev) == nil {
			out = append(out, &ev)
		}
	}
	return out, nil
}

func main() {
	cc, err := contractapi.NewChaincode(&TrustRegistry{})
	if err != nil {
		panic(err)
	}
	if err := cc.Start(); err != nil {
		panic(err)
	}
}
