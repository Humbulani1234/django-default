# Generated by Django 4.2.4 on 2023-10-02 11:36

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("logistic", "0003_item"),
    ]

    operations = [
        migrations.DeleteModel(
            name="Item",
        ),
    ]