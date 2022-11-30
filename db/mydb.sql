BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "person" (
	"id"	INTEGER NOT NULL,
	"name"	TEXT NOT NULL,
	"patronymic"	TEXT NOT NULL,
	"surname"	TEXT NOT NULL,
	"comment"	TEXT,
	"date_added"	TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "preprocessing" (
	"id"	INTEGER NOT NULL,
	"name"	TEXT NOT NULL,
	"comment"	TEXT,
	"date_added"	TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "model" (
	"id"	INTEGER NOT NULL,
	"name"	TEXT NOT NULL,
	"comment"	TEXT,
	"date_added"	TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "embedding" (
	"id"	INTEGER NOT NULL,
	"value"	TEXT NOT NULL,
	"date_added"	TEXT NOT NULL,
	"model_id"	INTEGER,
	"person_id"	INTEGER,
	"preprocessing_id"	INTEGER,
	FOREIGN KEY("preprocessing_id") REFERENCES "preprocessing"("id"),
	FOREIGN KEY("model_id") REFERENCES "model"("id"),
	FOREIGN KEY("person_id") REFERENCES "person"("id"),
	PRIMARY KEY("id" AUTOINCREMENT)
);
INSERT INTO "preprocessing" VALUES (1,'thumbnail',NULL,'2022-08-23');
INSERT INTO "model" VALUES (1,'tensorflow',NULL,'2022-08-23');
INSERT INTO "model" VALUES (2,'arcface',NULL,'2022-08-23');
COMMIT;
