CREATE TABLE IF NOT EXISTS supplies (
    id text PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS bidders (
    id text PRIMARY KEY,
    country text NOT NULL
);

CREATE TABLE IF NOT EXISTS supply_bidders (
    supply_id text NOT NULL REFERENCES supplies(id) ON DELETE CASCADE,
    bidder_id text NOT NULL REFERENCES bidders(id) ON DELETE CASCADE,
    PRIMARY KEY (supply_id, bidder_id)
);

CREATE INDEX IF NOT EXISTS idx_supply_bidders_supply_id ON supply_bidders (supply_id);
CREATE INDEX IF NOT EXISTS idx_bidders_country ON bidders (country);

INSERT INTO supplies (id)
VALUES ('supply1'), ('supply2')
ON CONFLICT (id) DO NOTHING;

INSERT INTO bidders (id, country)
VALUES
    ('bidder1', 'US'),
    ('bidder2', 'GB'),
    ('bidder3', 'US')
ON CONFLICT (id) DO NOTHING;

INSERT INTO supply_bidders (supply_id, bidder_id)
VALUES
    ('supply1', 'bidder1'),
    ('supply1', 'bidder2'),
    ('supply1', 'bidder3'),
    ('supply2', 'bidder2'),
    ('supply2', 'bidder3')
ON CONFLICT (supply_id, bidder_id) DO NOTHING;
